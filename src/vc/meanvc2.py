"""Stateful MeanVC2 worker on CPU or CUDA. No devices, training, downloads or reference reloads.

160ms ASR input hops feed pretrained 120ms+40ms VC blocks, optionally grouped
as in the human-approved long-phrase audition. Linear BN
interpolation is local in time; the offline whole-utterance interpolation is
noncausal. Streaming quality therefore needs its own listening verification.

`device='cpu'` reproduces the validated CPU route exactly. `device='cuda'` is the
same graph on the GPU: same weights, steps, masks, noise stream and float32
rounding (TF32 stays off; see service). Measured on a Kaggle Tesla T4 (2026-10-15,
tools/benchmark_meanvc2_realtime.py --devices cpu cuda, 20 s, six-block grouping):
the two routes agree to max_abs_diff 6.9e-4 and -76 dB, which is float32 rounding
rather than the same samples, and the GPU route is 2.0x faster (RTF 0.093 against
0.191 in the recorded run; 0.078 against 0.159 in the run before it, the shared VM
moves both). The recorded cpu/cuda clip pairs were also played back and settled by
the voice owner (2026-10-15, v6 material), which is what put meanvc2 into
CUDA_VERIFIED_BACKENDS. Moving a captured trace takes two steps: put it on the device before
`torch.jit.freeze` inlines the weights, then retarget the devices the graph itself
names (see retarget_devices). The kaldi frontend has no CUDA kernel, so filterbanks
are built on CPU and moved once per hop; only the final waveform crosses back, which
also synchronizes the stream. That frontend is what still dominates a GPU chunk:
9.3 ms of the 14.9 ms mean, against 4.8 ms for the VC blocks and 0.66 ms for Vocos.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np
from .base import VoiceConversionBackend
from ..runtime_paths import asset_root


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def retarget_devices(graphs, device):
    """Point every device a frozen trace names at where this run actually is.

    The capture machine's device gets baked into the graph as a constant: wenet's
    position encoder ends in `pe.to(device=cpu)`, so on a GPU the table stayed on
    CPU and the add that follows it failed with "found at least two devices".
    Neither `.to()` nor freeze reaches a graph constant, so the constant is
    rewritten. On the device these traces were captured on the values already
    match, so nothing moves and the validated CPU route is untouched.
    """
    target=str(device); moved=0
    for graph in graphs:
        for node in graph.findAllNodes('prim::Constant'):
            if str(node.output().type())!='Device':continue
            try:
                value=node.s('value')
            except RuntimeError:
                continue
            if value==target:continue
            node.s_('value',target);moved+=1
    return moved


class ReplayUnavailable(RuntimeError):
    """The captured stage did not reproduce the eager call; stay on the eager route."""


class OfflineAsrStage:
    """One ASR hop replayed from a captured CUDA graph.

    The frozen trace is four layers over 19 frames: on a GPU the arithmetic is
    trivial and the dispatch is not, which is why the features stage measured
    9.3 ms of a 14.9 ms chunk while the VC blocks took 4.8 ms. Capture records
    the same kernels once; replay writes the window and caches into the
    addresses the recording read and submits the stream.

    The trace reads the offset back to the host to index the position table,
    and a recording cannot contain a device readback, so the offset and the
    cache size are host scalars while the recording is made. That readback
    bakes the position window of the recording offset into the traced slices,
    so before each replay the stage copies the twelve rows the hop sits at
    into that window and restores the table afterwards; the arithmetic then
    reads the positions the eager call would have read. The call is still a
    pure function of window, offset and caches, so the recording is checked
    against the eager call at the recording offset and at two offsets it
    never saw; the eager route answers otherwise.

    Off CUDA there is nothing to record and the stage is just the same eager
    call reading the same buffers, which is what keeps the buffered schedule
    in `_offline_replay` under test on the CPU suite.
    """

    def __init__(self, backend, window_len):
        t=backend.torch;dev=backend.device
        self.backend=backend
        self.window=t.zeros(1,window_len,80,device=dev)
        # The route's first hop is never below 8 (reset and the position
        # rebase both start there) and the traced slice is empty below it.
        # The scalars stay on the host for the reason in the docstring; the
        # cache size is the route's eight cache frames.
        self.offset=t.full((),max(8,backend.asr_offset),dtype=t.int64)
        self.cache_size=t.full((),8,dtype=t.int64)
        self.att_cache=t.zeros_like(backend.att_cache)
        self.cnn_cache=t.zeros_like(backend.cnn_cache)
        self.bn,self.att_out,self.cnn_out=None,None,None
        self.baked_window=None
        self.graph=None
        if dev.type!='cuda':return
        with t.inference_mode():self.run()
        reference=(self.bn.clone(),self.att_out.clone(),self.cnn_out.clone())
        self.graph=backend._capture(self.run)
        baked=int(self.offset)
        self.baked_window=(baked-8,baked+4)
        self.graph.replay()
        for eager,replayed in zip(reference,(self.bn,self.att_out,self.cnn_out)):
            if not t.equal(eager,replayed):
                raise ReplayUnavailable('ASR replay differs from the eager call')
        # The route will ask for hops this recording never saw, up to the
        # rebase at 4996. The window swap has to carry them or the recording
        # is not trustworthy.
        for probe in (baked+4,baked+64):
            if probe<=4996:self._check(probe)

    def _check(self,offset):
        """The recorded hop against the eager call at a different offset."""
        t=self.backend.torch
        window=t.randn_like(self.window)
        att=t.randn_like(self.att_cache);cnn=t.randn_like(self.cnn_cache)
        with t.inference_mode():
            expected=self.backend.asr(window,t.tensor(offset,dtype=t.int64),self.cache_size,
                att.clone(),cnn.clone())
        replayed=self.call(window[0],offset,att,cnn)
        for eager,out in zip(expected,replayed):
            if not t.equal(eager,out):
                raise ReplayUnavailable('ASR replay does not track the offset')

    def run(self):
        self.bn,self.att_out,self.cnn_out=self.backend.asr(
            self.window,self.offset,self.cache_size,self.att_cache,self.cnn_cache)

    def call(self,window,offset,att_cache,cnn_cache):
        """Run one hop; ``None`` inputs keep whatever the buffers already hold."""
        if window is not None:self.window.copy_(window[None])
        if att_cache is not None:self.att_cache.copy_(att_cache)
        if cnn_cache is not None:self.cnn_cache.copy_(cnn_cache)
        if self.graph is None:
            self.offset.fill_(offset)
            self.run()
        else:
            self._load_window(offset)
            try:self.graph.replay()
            finally:self._restore_window()
        return self.bn,self.att_out,self.cnn_out

    def _load_window(self,offset):
        # The recorded slices read the rows the recording saw; the values are
        # the hop's, so the table carries them for the replay. Both sides are
        # cloned because the hop's window can overlap the recorded one.
        table=self.backend.position_table
        start,stop=self.baked_window
        rows=table[0,offset-8:offset+4].clone()
        self.table_rows=table[0,start:stop].clone()
        table[0,start:stop].copy_(rows)

    def _restore_window(self):
        table=self.backend.position_table
        start,stop=self.baked_window
        table[0,start:stop].copy_(self.table_rows)


class OfflineVcStage:
    """One VC step for a fixed group, replayed from a captured CUDA graph.

    The four-layer DiT is smaller than the dispatch that drives it: 3.7 ms per
    grouped call on a Tesla T4. The chunk mask caps attention at two past
    chunks (``t_p_num`` 2/2/1/1), which is the 24-frame cache the route keeps,
    so the captured group is the same computation the eager route ran. ``kv``
    is the cache the caller already holds, or ``None`` for the first group;
    the offset is that cache's length, exactly as the eager call computes it.
    """

    def __init__(self, backend, frames, step, kv):
        t=backend.torch;dev=backend.device
        self.backend=backend
        self.step=step
        self.x=t.zeros(1,frames,80,device=dev)
        self.cond=t.zeros(1,frames,256,device=dev)
        self.kv=None if kv is None else [(t.zeros_like(k),t.zeros_like(v)) for k,v in kv]
        self.offset=0 if kv is None else kv[0][0].shape[2]
        self.u=self.cache_out=None
        self.graph=None
        if dev.type!='cuda':return
        # The route builds stages inside `inference_mode`; a stage built
        # outside it can hit "Inference tensors cannot be saved for backward".
        with t.inference_mode():self.run()
        reference=(self.u.clone(),[(k.clone(),v.clone()) for k,v in self.cache_out])
        self.graph=backend._capture(self.run)
        self.graph.replay()
        if not t.equal(reference[0],self.u):
            raise ReplayUnavailable('VC replay differs from the eager call')
        for (eager_k,eager_v),(out_k,out_v) in zip(reference[1],self.cache_out):
            if not (t.equal(eager_k,out_k) and t.equal(eager_v,out_v)):
                raise ReplayUnavailable('VC replay caches differ from the eager call')

    def run(self):
        self.u,self.cache_out=self.backend.vc(self.x,*self.backend.timesteps[self.step],cache=None,
            cond=self.cond,spks=self.backend.speaker,offset=self.offset,is_inference=True,kv_cache=self.kv)

    def fill(self,x,cond,kv):
        """Load one group: the noised input, the conditions and the cache.

        The cache is part of the input, not just its shape; the stage is reused
        for every group with the same shapes, so the values are written in
        before each replay.
        """
        self.x.copy_(x);self.cond.copy_(cond)
        if self.kv is None:return
        for (target_k,target_v),(source_k,source_v) in zip(self.kv,kv):
            target_k.copy_(source_k);target_v.copy_(source_v)

    def step_once(self):
        """Run this step: assert the recording when there is one, else the call."""
        if self.graph is None:self.run()
        else:self.graph.replay()


class MeanVC2Backend(VoiceConversionBackend):
    chunk_samples=2560
    sample_rate=16000

    def __init__(self,threads=4,device='cpu'):
        if type(threads) is not int or not 1<=threads<=4:raise ValueError('CPU threads must be 1..4')
        if device not in ('cpu','cuda'):raise ValueError("MeanVC2 device must be 'cpu' or 'cuda'")
        self.threads=threads;self.device_requested=device;self.device=None
        self.vc=self.asr=self.vocos=None;self.stats={}
        self.traced_devices_moved=0
        self.vocoder_batch_frames=1
        self.vc_group_chunks=1
        self.steps=2
        # Captured stages per shape, keyed while loaded. Nothing to capture off
        # CUDA, where the same buffered schedule runs the calls eagerly.
        self.offline_stages=None
        self.offline_replay_enabled=True
        # Off CUDA the buffered schedule runs with its frontend thread beside
        # the solver; on CUDA the captured replay is the win and it stays serial.
        self.offline_overlap_enabled=True
        # Conditioning recipe: a profile choice set by select_profile.
        # 'streaming' is the shipped recipe, listened to and approved.
        self.offline_conditioning='streaming'
        self.replay_calls=0

    def load(self,model_path):
        started=time.perf_counter();folder=Path(model_path)
        runtime=json.loads((folder/'runtime.json').read_text('utf-8'))
        root=asset_root();repo=root/'vc_models/meanvc2/repo'
        if runtime['backend']!='meanvc2' or runtime['preset']!='120ms':raise ValueError('MeanVC2 120ms runtime required')
        for item in runtime['assets']:
            path=(root/item['path']).resolve()
            if not path.is_relative_to(root) or checksum(path)!=item['sha256']:
                raise ValueError('MeanVC2 asset checksum mismatch: '+item['path'])
        embedding=np.load(folder/'fixed_embedding.npy',allow_pickle=False)
        if embedding.shape!=(256,) or not np.isfinite(embedding).all():raise ValueError('Invalid fixed speaker embedding')
        import torch
        import torchaudio.compliance.kaldi as kaldi
        self.torch=torch;self.kaldi=kaldi
        if self.device_requested=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but no GPU is visible to the worker torch')
        self.device=torch.device('cuda' if self.device_requested=='cuda' else 'cpu')
        torch.set_num_threads(self.threads)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        # Import only inference modules; the app's src package remains intact.
        package=types.ModuleType('src.model');package.__path__=[str(repo/'src/model')]
        sys.modules['src.model']=package
        sys.path.insert(0,str(repo/'src/infer'))
        spec=importlib.util.spec_from_file_location('meanvc2_stream_upstream',repo/'src/infer/infer_e2e.py')
        official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        import modules_kvcahe
        def fused_attention(query,key,value,attn_mask=None,dropout_p=0.,is_causal=False,scale=None,enable_gqa=False):
            if attn_mask is not None:attn_mask=attn_mask[:,:,-query.shape[-2]:,:]
            return torch.nn.functional.scaled_dot_product_attention(query,key,value,attn_mask=attn_mask,
                dropout_p=dropout_p,is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
        modules_kvcahe.scaled_dot_product_attention_only=fused_attention
        cfg=json.loads((repo/'src/config/config_120ms_40ms.json').read_text('utf-8'))
        with (repo/'preprocess/ckpts/fastu2pp_160ms.pt').open('rb') as f:
            self.asr=torch.jit.load(f,map_location='cpu').eval()
        # Put the trace on its device before freezing it: freeze inlines the
        # weights as graph constants, and a .to() after that cannot reach them,
        # which is what made a CUDA run die on "weight type (torch.FloatTensor)".
        self.asr=torch.jit.freeze(self.asr.to(self.device),
                                  preserved_attrs=['m.encoder.embed.pos_enc.pe'])
        self.traced_devices_moved=retarget_devices([self.asr.graph],self.device)
        # The traced position encoder ends in pe.to(dtype,layout,device), with
        # the trace machine's device that retarget_devices rewrote above. The
        # table itself is a plain attribute: neither .to() nor freeze moves it
        # off CPU, so every hop would copy it to the compute device, which CUDA
        # graph capture refuses for an unpinned tensor. Move it once and bind
        # the route's table to the tensor the graph actually reads.
        pos_enc=self.asr.m.encoder.embed.pos_enc
        pos_enc.pe=pos_enc.pe.to(self.device)
        self.position_table=pos_enc.pe
        self.original_positions=self.position_table.clone()
        self.position_base=0;self.position_rolls=0
        self.position_frequencies=torch.exp(torch.arange(0,256,2,dtype=torch.float32,device=self.device)*(-np.log(10000.)/256))
        self.vc=official.DiT(**cfg['model']).float().eval()
        official.load_checkpoint(self.vc,str(repo/'ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors'),'cpu',use_ema=True)
        self.vc.to(self.device)
        with (repo/'ckpts/vocos/vocos.pt').open('rb') as f:
            self.vocos=torch.jit.load(f,map_location='cpu').eval()
        self.vocos=self.vocos.to(self.device)
        vocoder_shapes={k:list(v.shape) for k,v in self.vocos.state_dict().items() if v.ndim==3 and 'weight' in k}
        self.vocos=torch.jit.freeze(self.vocos,preserved_attrs=['decode'])
        self.traced_devices_moved+=retarget_devices([self.vocos.decode.graph],self.device)
        self.speaker=torch.from_numpy(embedding.copy()).float().unsqueeze(0).to(self.device)
        self.steps=runtime.get('steps',3)
        if type(self.steps) is not int or self.steps not in (2,3,4):raise ValueError('MeanVC2 steps must be 2, 3 or 4')
        self.feature_frontend=runtime.get('feature_frontend','legacy')
        self.bn_interpolation=runtime.get('bn_interpolation','legacy')
        if self.feature_frontend not in ('legacy','aligned'):raise ValueError('Invalid feature frontend')
        if self.bn_interpolation not in ('legacy','fixed_linear'):raise ValueError('Invalid BN interpolation')
        self.timesteps=[(torch.tensor([1-i/self.steps],device=self.device),torch.tensor([1-(i+1)/self.steps],device=self.device)) for i in range(self.steps)]
        self.cache_size_tensor=torch.tensor(8,device=self.device)
        self.interpolation_weights=torch.arange(4,dtype=torch.float32,device=self.device)[None,None,:,None]/4
        self.max_kv_frames=24 # Maximum attention history: two 120ms chunks.
        self.vocoder_left=runtime.get('vocoder_context',36);self.vocoder_right=self.vocoder_left
        if self.vocoder_left not in (32,36):raise ValueError('Vocoder context must be 32 or 36')
        self.vocoder_batch_frames=runtime.get('vocoder_batch_frames',1)
        if type(self.vocoder_batch_frames) is not int or self.vocoder_batch_frames not in (1,36):
            raise ValueError('Vocoder batch must be 1 or 36 frames')
        self.vc_group_chunks=runtime.get('vc_group_chunks',1)
        if type(self.vc_group_chunks) is not int or self.vc_group_chunks not in (1,3,6):
            raise ValueError('VC group must be 1, 3 or 6 chunks')
        # The buffered whole-file schedule runs everywhere; only the capture
        # behind it needs CUDA.
        self.offline_stages={}
        self.speaker_memory_forward=self.vc.gtm.forward
        with torch.inference_mode():
            memory=self.vc.gtm(self.speaker)
        # GTM depends only on the frozen speaker. Keep checkpoint weights intact.
        self.vc.gtm.forward=lambda _:memory
        self.reset()
        self.stats=dict(backend='MeanVC2 120ms / PyTorch '+('CUDA' if self.device.type=='cuda' else 'CPU'),name=runtime.get('display_name','MeanVC2 120ms · fixed reference'),
            sample_rate=16000,chunk_samples=2560,threads=self.threads,steps=self.steps,device=str(self.device),
            traced_devices_moved=self.traced_devices_moved,
            fixed_embedding_sha256=checksum(folder/'fixed_embedding.npy'),reference_sha256=checksum(folder/'reference.wav'),
            reference_fixed=True,prosody=False,text_aware=False,post_fx=False,
            attention_implementation='PyTorch scaled_dot_product_attention; same mask and FP32 weights',
            cache_max_frames=self.max_kv_frames,streaming_quality_verified=False,
            feature_frontend=self.feature_frontend,bn_interpolation=self.bn_interpolation,
            offline_conditioning=self.offline_conditioning,
            approved_variant=runtime.get('approved_variant'),human_approved_offline=runtime.get('human_approved_offline',False),
            interpolation_grid_delay_ms=40 if self.bn_interpolation=='fixed_linear' else 0,
            vocoder_batch_frames=self.vocoder_batch_frames,
            vc_group_chunks=self.vc_group_chunks,
            offline_replay=dict(used=False,captured=False,asr_graphs=0,vc_graphs=0,replays=0,reason=''),
            model_alignment_delay_ms=self.algorithmic_buffer_ms,algorithmic_buffer_ms=self.algorithmic_buffer_ms,
            load_seconds=time.perf_counter()-started)
        self.stats['vocoder_parameter_shapes']=vocoder_shapes

    @property
    def buffer_frames_ms(self):
        # The decoder's own delay line: whole input hops for grouped VC and
        # waveform bursts. reset() prefills pending_audio with exactly this many
        # samples, so this is the part grouping moves.
        base=640 if self.feature_frontend=='aligned' else 480
        extra=(self.vc_group_chunks-1)*12+self.vocoder_batch_frames-1
        return base+((extra+15)//16)*160

    @property
    def frontend_latency_ms(self):
        # Measured delay the frontend adds on top of that line, kept apart from it
        # because it is not buffer: correlating the input and output log-energy
        # envelopes (the same estimator returns a known 600ms shift exactly) gives
        # 50ms for the legacy frontend and exactly 0ms for the aligned one, at
        # both group6/vocoder36 and group1/vocoder1, so it does not scale with
        # grouping. It is the legacy branch's frame-origin carry (the 240-sample
        # wave_tail re-entering the fbank window).
        return 0 if self.feature_frontend=='aligned' else 50

    @property
    def algorithmic_buffer_ms(self):
        # The route's total input->output delay: what convert_utterance trims,
        # what the phrase repair aligns the source against, and what the GUI
        # reports. Treating it as the delay line alone left the trimmed result
        # 50ms late against the source on the legacy frontend, which cost the
        # retrospective shaping its alignment (measured energy_correlation 0.79
        # misaligned, 0.93 aligned). See notes/meanvc2_alignment.md.
        return self.buffer_frames_ms+self.frontend_latency_ms

    def _capture(self, run):
        """Record one stage call as a CUDA graph, ready to replay.

        Two warm-up calls on a side stream first, so the kernels exist and the
        allocator has settled before the recording starts; capture then sees a
        fixed set of addresses. Replay reads and writes those addresses, so the
        caller owns the static tensors and keeps them alive.
        """
        t=self.torch
        side=t.cuda.Stream()
        side.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(side), t.inference_mode():
            for _ in range(2):run()
        side.synchronize()
        graph=t.cuda.CUDAGraph()
        # The documented recipe: warm up and record on the same side stream, so
        # the recording never touches the default stream. Off that stream the
        # capture is rejected outright on some drivers.
        with t.inference_mode(), t.cuda.graph(graph, stream=side):
            run()
        return graph

    def _offline_stage(self, key, build):
        """The captured stage for one shape, or None when there will not be one.

        A stage that cannot be recorded is not a failure of the conversion: the
        offline route falls back to the eager call for that stage and keeps the
        other one captured. The reason is kept whole, because a capture refusal
        from the driver says why at the end of a long traceback.
        """
        if key in self.offline_stages:return self.offline_stages[key]
        if self.offline_stages is None:return None
        try:
            stage=build()
        except Exception as error:
            # ReplayUnavailable from the stage's own check reads the same here:
            # the recording exists but is not trusted, so do not use it.
            stage=None
            message=type(error).__name__+': '+str(error)
            # The driver's reason is the last line of a long TorchScript report,
            # so keep both ends of the message.
            if len(message)>2000:message=message[:400]+' ... '+message[-1600:]
            self.stats.setdefault('offline_replay',{}).update(**{key[0]+'_error':message})
        self.offline_stages[key]=stage
        return stage

    def _offline_asr(self, window_len):
        return self._offline_stage(('asr',window_len),lambda:OfflineAsrStage(self,window_len))

    def _offline_vc(self, frames, step, kv):
        return self._offline_stage(('vc',frames,step,0 if kv is None else kv[0][0].shape[2]),
                                   lambda:OfflineVcStage(self,frames,step,kv))

    def select_profile(self, selected, steps_allowed=(2,)):
        """Control/worker-side selection before warmup; reference stays fixed.

        ``steps_allowed`` is the caller's explicit budget: the shipped recipe is the
        two-step solver, and 3 and 4 measured RTF 1.130 and 1.343 (both over the
        realtime budget), so they are rejected rather than run. A one-step solver is
        allowed only where the caller asks for it by name -- the utterance route's
        fastest mode, which trades timbre accuracy for staying under RTF 1 -- and it
        clears the human-approved flags like any other recipe change.

        The quality route widens that budget and adds the offline recipe keys:
        ``feature_frontend`` ('legacy' | 'aligned': the streaming window
        construction or the upstream 19-frame grid) and ``offline_conditioning``
        ('streaming' | 'official': per-hop interpolation and noise or the
        upstream whole-utterance ones). 'official' needs the aligned frontend.
        Each is a recipe change, so it clears the same flags and waits on the
        same listening comparison.
        """
        if 'steps' in selected:
            steps = selected['steps']
            if type(steps) is not int or steps not in steps_allowed:
                allowed = '/'.join(str(value) for value in steps_allowed)
                raise ValueError(f'MeanVC2 supports the {allowed}-step solver only')
            if steps != self.steps:
                self.steps = steps
                self.timesteps = [(self.torch.tensor([1-i/steps],device=self.device), self.torch.tensor([1-(i+1)/steps],device=self.device))
                                  for i in range(steps)]
                if self.offline_stages is not None:self.offline_stages={}
                self.reset()
                self.stats['human_approved_offline'] = False
                self.stats['streaming_quality_verified'] = False
            self.stats['steps'] = steps
        # Inference grouping is a profile choice, not a checkpoint change. The
        # 120ms attention mask, KV cap, speaker conditioning and vocoder context
        # stay fixed; only how many blocks share one call changes, which moves
        # the fixed algorithmic buffer.
        changed = False
        for key, allowed in (('vc_group_chunks', (1, 3, 6)), ('vocoder_batch_frames', (1, 36))):
            # Grouping changes the group every captured stage was built for, so
            # the recordings are dropped with the profile that produced them.
            if key not in selected:
                continue
            value = selected[key]
            if type(value) is not int or value not in allowed:
                raise ValueError(f'Invalid MeanVC2 {key}')
            if getattr(self, key) != value:
                setattr(self, key, value)
                changed = True
        if selected.get('offline_conditioning', self.offline_conditioning) == 'official' \
                and selected.get('feature_frontend', self.feature_frontend) != 'aligned':
            raise ValueError('MeanVC2 official conditioning requires the aligned frontend')
        for key, allowed in (('feature_frontend', ('legacy', 'aligned')),
                             ('offline_conditioning', ('streaming', 'official'))):
            # The frontend moves the window grid and the delay line and the
            # conditioning changes what the solver sees; both invalidate
            # captured stages and approvals.
            if key not in selected:
                continue
            value = selected[key]
            if type(value) is not str or value not in allowed:
                raise ValueError(f'Invalid MeanVC2 {key}')
            if getattr(self, key) != value:
                setattr(self, key, value)
                changed = True
        if changed:
            if self.offline_stages is not None:self.offline_stages={}
            self.reset()
            self.stats['human_approved_offline'] = False
            self.stats['streaming_quality_verified'] = False
        if 'name' in selected:
            self.stats['name'] = selected['name']
        # The buffer statistics are published by load(); restate them only for a
        # loaded backend so an unloaded selection cannot read load-time state.
        if self.vc is not None:
            self.stats.update(vc_group_chunks=self.vc_group_chunks,
                              vocoder_batch_frames=self.vocoder_batch_frames,
                              feature_frontend=self.feature_frontend,
                              offline_conditioning=self.offline_conditioning,
                              algorithmic_buffer_ms=self.algorithmic_buffer_ms,
                              model_alignment_delay_ms=self.algorithmic_buffer_ms)
        if selected.get('optimized_condition'):
            self.stats['optimized_condition'] = selected['optimized_condition']

    def reset(self):
        if self.vc is None:return
        t=self.torch;dev=self.device
        self.att_cache=t.zeros(6,4,8,128,device=dev);self.cnn_cache=t.zeros(6,1,256,8,device=dev)
        self.asr_offset=8;self.fbank_history=t.zeros(3,80,device=dev)
        if self.position_base:
            self.position_table.copy_(self.original_positions)
            self.position_base=0
        self.wave_tail=np.zeros(240 if self.feature_frontend=='legacy' else 0,dtype=np.float32)
        self.fbank_pending=t.empty(0,80,device=dev)
        self.previous_bn=None;self.cond=t.empty(1,0,256,device=dev);self.noise=t.empty(1,0,80,device=dev)
        self.kv=None;self.mels=t.empty(1,80,0,device=dev);self.mel_origin=0;self.decoded_frames=0
        self.pending_audio=np.zeros(self.buffer_frames_ms*16,dtype=np.float32)
        # Draw the solver noise from a CPU generator whatever the compute device:
        # torch's CUDA generator is a different algorithm, so a GPU run would
        # otherwise see different numbers than the route that was listened to and
        # the two could never be compared. On CPU the copy below is a no-op.
        self.generator=t.Generator(device='cpu').manual_seed(110)
        self.calls=0

    def _extract_streaming_fbank(self,audio):
        t=self.torch
        wave=np.concatenate((self.wave_tail,audio))
        fbank=self.kaldi.fbank(t.from_numpy(wave*32768)[None],frame_length=25,frame_shift=10,
            snip_edges=True,num_mel_bins=80,energy_floor=0.,dither=0.,sample_frequency=16000)
        # The kaldi frontend has no CUDA kernel; move its small output once.
        fbank=fbank.to(self.device)
        if self.feature_frontend=='legacy':
            self.wave_tail=wave[-240:].copy()
            if len(fbank)!=16:raise RuntimeError('Unexpected streaming fbank hop')
        else:
            # Keep samples starting at the next 10ms frame origin. The first
            # hop has 14 complete frames; subsequent hops have 16. No synthetic
            # leading waveform/fbank frames enter the aligned ASR window.
            self.wave_tail=wave[len(fbank)*160:].copy()
        return fbank

    def _frontend_fragments(self,audio):
        """Run fbank + ASR for one 160ms hop; return ordered (cond, noise) frames.

        The frames are returned instead of being appended so that an overlapped
        variant can hand them to the conversion stage without sharing buffers.
        The arithmetic and the state written here are identical either way.
        """
        t=self.torch;fbank=self._extract_streaming_fbank(audio)
        if self.feature_frontend=='legacy':
            windows=[t.cat((self.fbank_history,fbank),dim=0)]
            self.fbank_history=windows[0][-3:].clone()
        else:
            self.fbank_pending=t.cat((self.fbank_pending,fbank),dim=0);windows=[]
            while len(self.fbank_pending)>=19:
                windows.append(self.fbank_pending[:19])
                self.fbank_pending=self.fbank_pending[16:]
        fragments=[]
        for window in windows:
            self._ensure_asr_positions()
            bn,self.att_cache,self.cnn_cache=self.asr(window[None],t.tensor(self.asr_offset,device=self.device),self.cache_size_tensor,self.att_cache,self.cnn_cache)
            self.asr_offset+=4
            fragments.append(self._condition_frames(bn))
        return fragments

    def _append_features(self,audio):
        for cond,noise in self._frontend_fragments(audio):
            self.cond=self.torch.cat((self.cond,cond),dim=1)
            self.noise=self.torch.cat((self.noise,noise),dim=1)

    def _condition_frames(self,bn):
        t=self.torch
        # Four 40ms BN frames become sixteen 10ms mel conditions. Carry one
        # preceding frame for interpolation continuity without future input.
        if self.bn_interpolation=='legacy':
            history=bn if self.previous_bn is None else t.cat((self.previous_bn,bn),dim=1)
            cond=t.nn.functional.interpolate(history.transpose(1,2),size=history.shape[1]*4,mode='linear',align_corners=True).transpose(1,2)
            if self.previous_bn is not None:cond=cond[:,4:]
        else:
            previous=bn[:,:1] if self.previous_bn is None else self.previous_bn
            history=t.cat((previous,bn),dim=1)
            left=history[:,:-1,None,:];right=history[:,1:,None,:]
            cond=(left+(right-left)*self.interpolation_weights).reshape(1,-1,256)
        self.previous_bn=bn[:,-1:].clone()
        return cond,t.randn(1,cond.shape[1],80,generator=self.generator).to(self.device)

    def _append_bn(self,bn):
        cond,noise=self._condition_frames(bn)
        self.cond=self.torch.cat((self.cond,cond),dim=1)
        self.noise=self.torch.cat((self.noise,noise),dim=1)

    def _ensure_asr_positions(self):
        # The traced ASR has only 5000 sine/cosine positions (~200 seconds).
        # Slide this fixed-size table, retaining absolute sinusoidal phases
        # and all neural caches. Only its local slice index is rebased.
        if self.asr_offset+4<=self.position_table.shape[1]:return
        self.position_base+=self.asr_offset-8
        self.asr_offset=8
        t=self.torch
        positions=t.arange(self.position_table.shape[1],dtype=t.float32,device=self.device)+self.position_base
        phase=positions[:,None]*self.position_frequencies[None,:]
        self.position_table[0,:,0::2].copy_(t.sin(phase))
        self.position_table[0,:,1::2].copy_(t.cos(phase))
        self.position_rolls+=1

    def _convert_blocks(self):
        t=self.torch
        emitted=self.vc_group_chunks*12
        required=emitted+4
        while self.cond.shape[1]>=required:
            cond=self.cond[:,:required];x=self.noise[:,:required].clone()
            # Keys in the upstream cache are unrotated. Rebasing by a whole
            # chunk preserves relative RoPE positions and chunk-mask phase.
            offset=0 if self.kv is None else self.kv[0][0].shape[2]
            for step in range(self.steps):
                a=1-step/self.steps;b=1-(step+1)/self.steps
                u,cache=self.vc(x,*self.timesteps[step],cache=None,cond=cond,spks=self.speaker,
                                offset=offset,is_inference=True,kv_cache=self.kv)
                x=x-(a-b)*u
            self.kv=[(k[:,:,-self.max_kv_frames:].contiguous(),v[:,:,-self.max_kv_frames:].contiguous()) for k,v in cache]
            self.mels=t.cat((self.mels,(x[:,:emitted].transpose(1,2)+1)/2),dim=2)
            self.cond=self.cond[:,emitted:];self.noise=self.noise[:,emitted:]

    def _decode_ready(self):
        # Streaming cadence: only decode once a whole batch is stable.
        total=self.mel_origin+self.mels.shape[2]
        end=total-self.vocoder_right
        if end-self.decoded_frames<self.vocoder_batch_frames:return
        self._decode_stable(end)

    def _decode_stable(self,end):
        """Decode the retained mel frames and keep only the stable samples.

        Nine width-7 convolutions give 27-frame context per side, plus the
        ISTFT window. Retain 36 frames on both sides and crop only stable
        samples. Work and RAM do not grow with utterance. `end` is the
        exclusive frame index the caller has decided is ready: the streaming
        route waits for a full batch, the offline one has the whole file and
        passes everything it retains.
        """
        if end<=self.decoded_frames:return False
        decoded=self.vocos.decode(self.mels).squeeze().cpu().numpy()
        start=(self.decoded_frames-self.mel_origin)*160;stop=(end-self.mel_origin)*160
        if len(decoded)<stop:raise RuntimeError('Unexpected Vocos sample count')
        self.pending_audio=np.concatenate((self.pending_audio,decoded[start:stop].astype(np.float32)))
        self.decoded_frames=end
        discard=max(0,end-self.vocoder_left-self.mel_origin)
        if discard:self.mels=self.mels[:,:,discard:].clone();self.mel_origin+=discard
        return True

    def process_chunk(self,audio):
        if self.vc is None:raise RuntimeError('MeanVC2 is not loaded')
        if audio.shape!=(2560,) or audio.dtype!=np.float32 or not np.isfinite(audio).all():
            raise ValueError('MeanVC2 expects 160ms finite float32 mono')
        with self.torch.inference_mode():
            begin=time.perf_counter();self._append_features(audio);features=time.perf_counter()
            self._convert_blocks();convert=time.perf_counter();self._decode_ready();finish=time.perf_counter()
        self.stats['last_stage_ms']=dict(features=(features-begin)*1000,vc=(convert-features)*1000,vocoder=(finish-convert)*1000)
        if len(self.pending_audio)<2560:raise RuntimeError('MeanVC2 streaming output alignment exhausted')
        result=self.pending_audio[:2560].copy();self.pending_audio=self.pending_audio[2560:]
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite MeanVC2 output')
        self.calls+=1
        return result

    def _fbank_whole(self,wave):
        """One kaldi call for a whole waveform; the same frames, frame for frame.

        The per-hop loop rebuilds 16 frames per call from the same grid, so the
        values agree to float32 rounding (measured max_abs_diff 1.9e-06 on the
        log-mel frames) while the call itself is ten times cheaper. The grid
        offset the legacy frontend needs is the caller's: it feeds
        ``zeros(240)+audio``, which is exactly the carry its first window built.
        """
        frames=self.kaldi.fbank(self.torch.from_numpy(wave*32768)[None],frame_length=25,frame_shift=10,
            snip_edges=True,num_mel_bins=80,energy_floor=0.,dither=0.,sample_frequency=16000)
        return frames.to(self.device)

    def _frontend_windows(self,frames,hops):
        """The windows the streaming frontend would have built, in order.

        Only the window construction is replayed; the ASR still consumes them
        one call at a time, because batching hops into one call changes the
        subsampling boundary and the measured output (max_abs_diff 4.6 at two
        hops, growing with the batch).
        """
        t=self.torch
        if self.feature_frontend=='legacy':
            history=t.zeros(3,80,device=self.device)
            for hop in range(hops):
                window=t.cat((history,frames[16*hop:16*hop+16]),dim=0)
                history=window[-3:].clone()
                yield window
            return
        pending=t.empty(0,80,device=self.device);cursor=0
        for hop in range(hops):
            # Frames fully inside the samples the stream has consumed by now.
            stop=max(0,(self.chunk_samples*(hop+1)-400)//160+1)
            pending=t.cat((pending,frames[cursor:stop]),dim=0);cursor=stop
            while pending.shape[0]>=19:
                yield pending[:19]
                pending=pending[16:]

    def convert_offline(self,audio,replay=None):
        """Convert one recorded waveform in a single pass.

        Same neural schedule as the streaming route -- one 160ms ASR window per
        hop, the same windows, grouping, attention mask and vocoder context --
        and the same frame stream the streaming route would have emitted.
        Scheduling is what changes: the filterbank runs once for the whole
        input, mel frames stay on the compute device, and the decoded waveform
        crosses back once instead of every hop. On a GPU that is the difference
        between waiting for a per-hop round trip and keeping the device busy.

        ``audio`` is the 16kHz input the hop loop would consume, zero-padded to
        a whole number of hops if it is not one already; the return value is the
        same concatenated stream ``process_chunk`` would have returned, so the
        caller trims the same delay.

        On CUDA the two heavy stages are replayed from captured graphs by
        default, which is the difference between the arithmetic and the
        dispatch that drives it. ``replay=False`` forces the eager loop, and a
        capture that fails its own check falls back to it with the reason in
        the ``offline_replay`` statistics. The buffered schedule itself runs on
        both devices; off CUDA its frontend thread runs beside the solver,
        handing the same buffers over at the same cursor, so the samples are
        the serial route's and the pass is shorter.

        ``offline_conditioning='official'`` runs the upstream script's own
        whole-file schedule instead: every window is seen first, the conditions
        are interpolated once for the whole utterance and the solver noise is
        drawn in one call, as the upstream inference does. Both need the
        frontend finished, so that route cannot overlap it. It is the quality
        comparison variant, not the shipped route.
        """
        if self.vc is None:raise RuntimeError('MeanVC2 is not loaded')
        audio=np.ascontiguousarray(audio)
        if audio.ndim!=1 or audio.dtype!=np.float32 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError('MeanVC2 offline conversion expects finite float32 mono')
        hop=self.chunk_samples;hops=(len(audio)+hop-1)//hop;samples=hops*hop
        if len(audio)%hop:
            audio=np.concatenate((audio,np.zeros(samples-len(audio),dtype=np.float32)))
        self.reset()
        started=time.perf_counter()
        if self.feature_frontend=='legacy':
            # The carry the first streaming window built, in one waveform.
            wave=np.concatenate((np.zeros(240,dtype=np.float32),audio))
        else:
            wave=audio
        frames=self._fbank_whole(wave)
        if replay is None:replay=self.offline_replay_enabled
        if self.offline_conditioning=='official':
            if self.feature_frontend!='aligned':
                raise ValueError('MeanVC2 official conditioning requires the aligned frontend')
            # Captured stages still replay; only the overlapped schedule is
            # lost, because the conditions are not final until every window
            # has been seen.
            return self._offline_official(frames,hops,samples,started,
                use_stages=bool(replay and self.offline_stages is not None))
        if replay and self.offline_stages is not None:
            try:
                # Off CUDA the frontend thread runs the buffered schedule beside
                # the solver; on CUDA the graphs are the win and it stays serial.
                if self.device.type=='cpu' and self.offline_overlap_enabled:
                    return self._offline_overlap(frames,hops,samples,started)
                return self._offline_replay(frames,hops,samples,started)
            except ReplayUnavailable as error:
                self.stats['offline_replay']=dict(used=False,captured=False,asr_graphs=0,vc_graphs=0,
                    replays=self.replay_calls,reason=str(error)[:300])
                self.reset()
        return self._offline_eager(frames,hops,samples,started)

    def _offline_stage_stats(self,used):
        stage=lambda kind:[value for key,value in self.offline_stages.items()
                           if key[0]==kind and value is not None]
        errors={key:value for key,value in self.stats.get('offline_replay',{}).items()
                if key.endswith('_error')}
        self.stats['offline_replay']=dict(used=used,
            captured=self.device.type=='cuda' and any(key[0] in ('asr','vc')
                     and value is not None for key,value in self.offline_stages.items()),
            asr_graphs=len(stage('asr')),vc_graphs=len(stage('vc')),
            replays=self.replay_calls,reason='',**errors)

    def _offline_output(self,samples):
        if len(self.pending_audio)<samples:raise RuntimeError('MeanVC2 streaming output alignment exhausted')
        result=self.pending_audio[:samples].copy()
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite MeanVC2 output')
        return result

    def _offline_eager(self,frames,hops,samples,started):
        """The hop-by-hop schedule: one dispatched call per stage, per hop."""
        t=self.torch
        features=vc=0.
        with t.inference_mode():
            for window in self._frontend_windows(frames,hops):
                self._ensure_asr_positions()
                begin=time.perf_counter()
                bn,self.att_cache,self.cnn_cache=self.asr(window[None],
                    t.tensor(self.asr_offset,device=self.device),self.cache_size_tensor,
                    self.att_cache,self.cnn_cache)
                self.asr_offset+=4
                cond,noise=self._condition_frames(bn)
                self.cond=t.cat((self.cond,cond),dim=1);self.noise=t.cat((self.noise,noise),dim=1)
                features+=time.perf_counter()-begin
                begin=time.perf_counter()
                self._convert_blocks()
                vc+=time.perf_counter()-begin
            begin=time.perf_counter()
            self._decode_stable(self.mel_origin+self.mels.shape[2]-self.vocoder_right)
            vocoder=time.perf_counter()-begin
        elapsed=time.perf_counter()-started
        result=self._offline_output(samples)
        self.stats['offline_stage_seconds']=dict(features=features,vc=vc,vocoder=vocoder,
            fbank=elapsed-features-vc-vocoder)
        self.stats['offline_hops']=hops
        self.stats['offline_input_samples']=samples
        self.calls+=hops
        return result

    def _offline_replay(self,frames,hops,samples,started):
        """The same schedule with both heavy stages replayed from capture.

        The frames the streaming route would have built still come from the same
        windows in the same order, and the solver still runs group by group with
        a 24-frame attention cache, so the frame stream is the eager route's.
        What is gone is the per-hop dispatch: the window, offset and caches are
        written into the addresses the recording read, and one replay submits
        the recorded stream instead of the four layers being driven from Python.
        """
        t=self.torch;dev=self.device
        chunk=self.vc_group_chunks*12;required=chunk+4
        cond_buffer=t.empty(1,16*hops,256,device=dev)
        noise_buffer=t.empty(1,16*hops,80,device=dev)
        cursor=0;features=0.
        with t.inference_mode():
            for window in self._frontend_windows(frames,hops):
                self._ensure_asr_positions()
                begin=time.perf_counter()
                stage=self._offline_asr(window.shape[0])
                if stage is None:
                    bn,self.att_cache,self.cnn_cache=self.asr(window[None],
                        t.tensor(self.asr_offset,device=self.device),self.cache_size_tensor,
                        self.att_cache,self.cnn_cache)
                else:
                    bn,self.att_cache,self.cnn_cache=stage.call(window,self.asr_offset,
                        self.att_cache,self.cnn_cache)
                self.asr_offset+=4
                cond,noise=self._condition_frames(bn)
                span=cond.shape[1]
                if cursor+span>cond_buffer.shape[1]:
                    raise ReplayUnavailable('More frontend windows than hops')
                cond_buffer[:,cursor:cursor+span]=cond
                noise_buffer[:,cursor:cursor+span]=noise
                cursor+=span
                features+=time.perf_counter()-begin
            groups=(cursor-4)//chunk if cursor>=required else 0
            self.mels=t.empty(1,80,groups*chunk,device=dev)
            vc=0.
            for group in range(groups):
                begin_time=time.perf_counter()
                begin=group*chunk
                x=noise_buffer[:,begin:begin+required].clone()
                cond=cond_buffer[:,begin:begin+required]
                last_cache=None
                for step in range(self.steps):
                    a=1-step/self.steps;b=1-(step+1)/self.steps
                    stage=self._offline_vc(required,step,self.kv)
                    if stage is None:
                        offset=0 if self.kv is None else self.kv[0][0].shape[2]
                        u,last_cache=self.vc(x,*self.timesteps[step],cache=None,cond=cond,
                            spks=self.speaker,offset=offset,is_inference=True,kv_cache=self.kv)
                    else:
                        stage.fill(x,cond,self.kv)
                        stage.step_once()
                        if stage.graph is not None:self.replay_calls+=1
                        u,last_cache=stage.u,stage.cache_out
                    x=x-(a-b)*u
                self.kv=[(k[:,:,-self.max_kv_frames:].contiguous(),v[:,:,-self.max_kv_frames:].contiguous())
                         for k,v in last_cache]
                self.mels[:,:,begin:begin+chunk]=(x[:,:chunk].transpose(1,2)+1)/2
                vc+=time.perf_counter()-begin_time
            begin=time.perf_counter()
            self._decode_stable(self.mel_origin+self.mels.shape[2]-self.vocoder_right)
            vocoder=time.perf_counter()-begin
        elapsed=time.perf_counter()-started
        result=self._offline_output(samples)
        self.att_cache=self.att_cache.clone();self.cnn_cache=self.cnn_cache.clone()
        self.stats['offline_stage_seconds']=dict(features=features,vc=vc,vocoder=vocoder,
            fbank=elapsed-features-vc-vocoder)
        self.stats['offline_hops']=hops
        self.stats['offline_input_samples']=samples
        self._offline_stage_stats(True)
        self.calls+=hops
        return result

    def _offline_official(self,frames,hops,samples,started,use_stages):
        """The upstream script's whole-file schedule: all conditions first.

        The upstream inference interpolates the BN frames once for the whole
        utterance and draws the solver noise the same way, so nothing converts
        until every window has been seen: the frontend runs to the end, then
        the groups. Same windows, grouping, mask, KV cap, vocoder context and
        trim as the other offline routes; what the streaming route cannot
        express is an interpolation whose neighbours span hop boundaries and a
        noise draw that does not slice per hop. Giving up the overlapped
        frontend thread is the cost of the schedule on CPU. The recipe waits on
        a listening comparison; the shipped route keeps streaming conditioning.
        """
        t=self.torch;dev=self.device
        chunk=self.vc_group_chunks*12;required=chunk+4
        bn_buffer=t.empty(1,4*hops,256,device=dev)
        cursor=0;features=0.
        with t.inference_mode():
            for window in self._frontend_windows(frames,hops):
                self._ensure_asr_positions()
                begin=time.perf_counter()
                stage=self._offline_asr(window.shape[0]) if use_stages else None
                if stage is None:
                    bn,self.att_cache,self.cnn_cache=self.asr(window[None],
                        t.tensor(self.asr_offset,device=dev),self.cache_size_tensor,
                        self.att_cache,self.cnn_cache)
                else:
                    bn,self.att_cache,self.cnn_cache=stage.call(window,self.asr_offset,
                        self.att_cache,self.cnn_cache)
                self.asr_offset+=4
                span=bn.shape[1]
                if cursor+span>bn_buffer.shape[1]:
                    raise ReplayUnavailable('More frontend windows than hops')
                bn_buffer[:,cursor:cursor+span]=bn
                cursor+=span
                features+=time.perf_counter()-begin
            cond=t.nn.functional.interpolate(bn_buffer[:,:cursor].transpose(1,2),size=cursor*4,
                mode='linear',align_corners=True).transpose(1,2)
            noise=t.randn(1,cursor*4,80,generator=self.generator).to(dev)
            mel_frames=cursor*4
            groups=(mel_frames-4)//chunk if mel_frames>=required else 0
            self.mels=t.empty(1,80,groups*chunk,device=dev)
            vc=0.
            for group in range(groups):
                begin_time=time.perf_counter()
                begin=group*chunk
                x=noise[:,begin:begin+required].clone()
                cond_in=cond[:,begin:begin+required]
                last_cache=None
                for step in range(self.steps):
                    a=1-step/self.steps;b=1-(step+1)/self.steps
                    stage=self._offline_vc(required,step,self.kv) if use_stages else None
                    if stage is None:
                        offset=0 if self.kv is None else self.kv[0][0].shape[2]
                        u,last_cache=self.vc(x,*self.timesteps[step],cache=None,cond=cond_in,
                            spks=self.speaker,offset=offset,is_inference=True,kv_cache=self.kv)
                    else:
                        stage.fill(x,cond_in,self.kv)
                        stage.step_once()
                        if stage.graph is not None:self.replay_calls+=1
                        u,last_cache=stage.u,stage.cache_out
                    x=x-(a-b)*u
                self.kv=[(k[:,:,-self.max_kv_frames:].contiguous(),v[:,:,-self.max_kv_frames:].contiguous())
                         for k,v in last_cache]
                self.mels[:,:,begin:begin+chunk]=(x[:,:chunk].transpose(1,2)+1)/2
                vc+=time.perf_counter()-begin_time
            begin=time.perf_counter()
            self._decode_stable(self.mel_origin+self.mels.shape[2]-self.vocoder_right)
            vocoder=time.perf_counter()-begin
        elapsed=time.perf_counter()-started
        result=self._offline_output(samples)
        self.att_cache=self.att_cache.clone();self.cnn_cache=self.cnn_cache.clone()
        self.stats['offline_stage_seconds']=dict(features=features,vc=vc,vocoder=vocoder,
            fbank=elapsed-features-vc-vocoder)
        self.stats['offline_hops']=hops
        self.stats['offline_input_samples']=samples
        self._offline_stage_stats(bool(use_stages))
        self.stats['offline_replay']['official']=True
        self.calls+=hops
        return result

    def _offline_overlap(self,frames,hops,samples,started):
        """The buffered schedule with the frontend thread running beside it.

        The two stages own disjoint state -- the frontend writes the ASR caches,
        the position table and ``previous_bn``, the solver writes ``kv``,
        ``mels`` and the noise generator -- so the condition and noise buffers
        are the whole hand-off, and the cursor is the whole protocol: the
        frontend appends one hop's frames and bumps it, the solver converts the
        groups below it. Same windows, same buffers and the same group
        boundaries as ``_offline_replay``, so the samples are that route's bit
        for bit; what changes is that the frontend is already ahead when a
        group falls due, which is most of the pass on a four-core CPU.
        """
        t=self.torch;dev=self.device
        chunk=self.vc_group_chunks*12;required=chunk+4
        cond_buffer=t.empty(1,16*hops,256,device=dev)
        noise_buffer=t.empty(1,16*hops,80,device=dev)
        capacity=(16*hops-4)//chunk
        state=dict(cursor=0,features=0.,produced=False,failure=None,wait=0.)
        ready=threading.Condition()

        def produce():
            features=0.
            try:
                with t.inference_mode():
                    for window in self._frontend_windows(frames,hops):
                        self._ensure_asr_positions()
                        begin=time.perf_counter()
                        stage=self._offline_asr(window.shape[0])
                        if stage is None:
                            bn,self.att_cache,self.cnn_cache=self.asr(window[None],
                                t.tensor(self.asr_offset,device=dev),self.cache_size_tensor,
                                self.att_cache,self.cnn_cache)
                        else:
                            bn,self.att_cache,self.cnn_cache=stage.call(window,self.asr_offset,
                                self.att_cache,self.cnn_cache)
                        self.asr_offset+=4
                        cond,noise=self._condition_frames(bn)
                        span=cond.shape[1]
                        if state['cursor']+span>cond_buffer.shape[1]:
                            raise ReplayUnavailable('More frontend windows than hops')
                        cond_buffer[:,state['cursor']:state['cursor']+span]=cond
                        noise_buffer[:,state['cursor']:state['cursor']+span]=noise
                        # The buffers are written before the cursor moves, so a
                        # waiter woken by this bump can read everything below it.
                        with ready:
                            state['cursor']+=span
                            ready.notify_all()
                        features+=time.perf_counter()-begin
            except BaseException as error:
                state['failure']=error
            finally:
                with ready:
                    state['features']=features
                    state['produced']=True
                    ready.notify_all()

        worker=threading.Thread(target=produce,name='meanvc2-offline-frontend',daemon=True)
        worker.start()
        vc=0.;vocoder=0.
        try:
            with t.inference_mode():
                # The buffer holds the most groups a hop's worth of windows can
                # ever make; the frontend may build fewer (the aligned grid can
                # end short), so the frames past the last group are cut before
                # the decoder sees them.
                self.mels=t.empty(1,80,capacity*chunk,device=dev)
                group=0
                while True:
                    begin=group*chunk;needed=begin+required
                    waited=time.perf_counter()
                    with ready:
                        while state['cursor']<needed and not state['produced'] and state['failure'] is None:
                            ready.wait(0.02)
                        state['wait']+=time.perf_counter()-waited
                        due=state['cursor']>=needed
                    if state['failure'] is not None:raise state['failure']
                    if not due:break
                    group_started=time.perf_counter()
                    x=noise_buffer[:,begin:begin+required].clone()
                    cond=cond_buffer[:,begin:begin+required]
                    last_cache=None
                    for step in range(self.steps):
                        a=1-step/self.steps;b=1-(step+1)/self.steps
                        stage=self._offline_vc(required,step,self.kv)
                        if stage is None:
                            offset=0 if self.kv is None else self.kv[0][0].shape[2]
                            u,last_cache=self.vc(x,*self.timesteps[step],cache=None,cond=cond,
                                spks=self.speaker,offset=offset,is_inference=True,kv_cache=self.kv)
                        else:
                            stage.fill(x,cond,self.kv)
                            stage.step_once()
                            if stage.graph is not None:self.replay_calls+=1
                            u,last_cache=stage.u,stage.cache_out
                        x=x-(a-b)*u
                    self.kv=[(k[:,:,-self.max_kv_frames:].contiguous(),v[:,:,-self.max_kv_frames:].contiguous())
                             for k,v in last_cache]
                    self.mels[:,:,begin:begin+chunk]=(x[:,:chunk].transpose(1,2)+1)/2
                    vc+=time.perf_counter()-group_started
                    group+=1
                if group*chunk!=self.mels.shape[2]:
                    self.mels=self.mels[:,:,:group*chunk].contiguous()
                decoder=time.perf_counter()
                self._decode_stable(self.mel_origin+self.mels.shape[2]-self.vocoder_right)
                vocoder=time.perf_counter()-decoder
        finally:
            worker.join()
        elapsed=time.perf_counter()-started
        result=self._offline_output(samples)
        self.att_cache=self.att_cache.clone();self.cnn_cache=self.cnn_cache.clone()
        # The stage numbers stay the stage numbers: `features` is the frontend
        # thread's busy time (it overlaps the solver, so the sum now exceeds the
        # wall), and the residual is what neither thread held.
        self.stats['offline_stage_seconds']=dict(features=state['features'],vc=vc,vocoder=vocoder,
            fbank=max(0.,elapsed-max(state['features'],vc)-vocoder))
        self.stats['offline_overlap']=dict(frontend_seconds=state['features'],solver_seconds=vc,
            wait_seconds=state['wait'])
        self.stats['offline_hops']=hops
        self.stats['offline_input_samples']=samples
        self._offline_stage_stats(True)
        self.stats['offline_replay']['overlap']=True
        self.calls+=hops
        return result

    def warmup(self):
        started=time.perf_counter()
        # Grouped decoding must reach its first real Vocos call during startup,
        # rather than compiling that path on the first spoken sentence.
        count=4 if self.vocoder_batch_frames==1 and self.vc_group_chunks==1 else (self.algorithmic_buffer_ms+159)//160+self.vc_group_chunks+1
        for _ in range(count):self.process_chunk(np.zeros(2560,dtype=np.float32))
        self.stats['warmup_seconds']=time.perf_counter()-started;self.reset()

    def unload(self):
        self.vc=self.asr=self.vocos=None;self.kv=None
        if self.offline_stages is not None:self.offline_stages={}

    def get_stats(self):return dict(self.stats,asr_position_rolls=self.position_rolls,asr_position_base=self.position_base)
