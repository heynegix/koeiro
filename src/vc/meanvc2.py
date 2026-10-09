"""Stateful MeanVC2 worker on CPU or CUDA. No devices, training, downloads or reference reloads.

160ms ASR input hops feed pretrained 120ms+40ms VC blocks, optionally grouped
as in the human-approved long-phrase audition. Linear BN
interpolation is local in time; the offline whole-utterance interpolation is
noncausal. Streaming quality therefore needs its own listening verification.

`device='cpu'` reproduces the validated CPU route exactly. `device='cuda'`
runs the same float32 graph on the GPU: same weights, steps, masks and
float32 rounding (TF32 stays off; see service). The kaldi frontend has no CUDA
kernel, so filterbanks are built on CPU and moved once per hop; only the
final waveform crosses back, which also synchronizes the stream.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
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


class MeanVC2Backend(VoiceConversionBackend):
    chunk_samples=2560
    sample_rate=16000

    def __init__(self,threads=4,device='cpu'):
        if type(threads) is not int or not 1<=threads<=4:raise ValueError('CPU threads must be 1..4')
        if device not in ('cpu','cuda'):raise ValueError("MeanVC2 device must be 'cpu' or 'cuda'")
        self.threads=threads;self.device_requested=device;self.device=None
        self.vc=self.asr=self.vocos=None;self.stats={}
        self.vocoder_batch_frames=1
        self.vc_group_chunks=1
        self.steps=2

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
        self.asr=torch.jit.freeze(self.asr,preserved_attrs=['m.encoder.embed.pos_enc.pe']).to(self.device)
        self.position_table=self.asr.m.encoder.embed.pos_enc.pe
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
        self.speaker_memory_forward=self.vc.gtm.forward
        with torch.inference_mode():
            memory=self.vc.gtm(self.speaker)
        # GTM depends only on the frozen speaker. Keep checkpoint weights intact.
        self.vc.gtm.forward=lambda _:memory
        self.reset()
        self.stats=dict(backend='MeanVC2 120ms / PyTorch '+('CUDA' if self.device.type=='cuda' else 'CPU'),name=runtime.get('display_name','MeanVC2 120ms · fixed reference'),
            sample_rate=16000,chunk_samples=2560,threads=self.threads,steps=self.steps,device=str(self.device),
            fixed_embedding_sha256=checksum(folder/'fixed_embedding.npy'),reference_sha256=checksum(folder/'reference.wav'),
            reference_fixed=True,prosody=False,text_aware=False,post_fx=False,
            attention_implementation='PyTorch scaled_dot_product_attention; same mask and FP32 weights',
            cache_max_frames=self.max_kv_frames,streaming_quality_verified=False,
            feature_frontend=self.feature_frontend,bn_interpolation=self.bn_interpolation,
            approved_variant=runtime.get('approved_variant'),human_approved_offline=runtime.get('human_approved_offline',False),
            interpolation_grid_delay_ms=40 if self.bn_interpolation=='fixed_linear' else 0,
            vocoder_batch_frames=self.vocoder_batch_frames,
            vc_group_chunks=self.vc_group_chunks,
            model_alignment_delay_ms=None,algorithmic_buffer_ms=self.algorithmic_buffer_ms,
            load_seconds=time.perf_counter()-started)
        self.stats['vocoder_parameter_shapes']=vocoder_shapes

    @property
    def algorithmic_buffer_ms(self):
        # Reserve whole input hops for both grouped VC and waveform bursts.
        base=640 if self.feature_frontend=='aligned' else 480
        extra=(self.vc_group_chunks-1)*12+self.vocoder_batch_frames-1
        return base+((extra+15)//16)*160

    def select_profile(self, selected):
        """Control/worker-side selection before warmup; reference stays fixed."""
        if 'steps' in selected:
            steps = selected['steps']
            # Only the two-step solver ships. 3 and 4 measured RTF 1.130 and 1.343,
            # both over the realtime budget, so they are rejected rather than run.
            if type(steps) is not int or steps != 2:
                raise ValueError('MeanVC2 supports the 2-step solver only')
            if steps != self.steps:
                self.steps = steps
                self.timesteps = [(self.torch.tensor([1-i/steps],device=self.device), self.torch.tensor([1-(i+1)/steps],device=self.device))
                                  for i in range(steps)]
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
            if key not in selected:
                continue
            value = selected[key]
            if type(value) is not int or value not in allowed:
                raise ValueError(f'Invalid MeanVC2 {key}')
            if getattr(self, key) != value:
                setattr(self, key, value)
                changed = True
        if changed:
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
                              algorithmic_buffer_ms=self.algorithmic_buffer_ms)
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
        self.pending_audio=np.zeros(self.algorithmic_buffer_ms*16,dtype=np.float32)
        self.generator=t.Generator(device=dev).manual_seed(110)
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
        return cond,t.randn(1,cond.shape[1],80,generator=self.generator,device=self.device)

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
        # Nine width-7 convolutions give 27-frame context per side, plus the
        # ISTFT window. Retain 36 frames on both sides and crop only stable
        # samples. Work and RAM do not grow with utterance.
        total=self.mel_origin+self.mels.shape[2]
        end=total-self.vocoder_right
        if end-self.decoded_frames<self.vocoder_batch_frames:return
        decoded=self.vocos.decode(self.mels).squeeze().cpu().numpy()
        start=(self.decoded_frames-self.mel_origin)*160;stop=(end-self.mel_origin)*160
        if len(decoded)<stop:raise RuntimeError('Unexpected Vocos sample count')
        self.pending_audio=np.concatenate((self.pending_audio,decoded[start:stop].astype(np.float32)))
        self.decoded_frames=end
        discard=max(0,end-self.vocoder_left-self.mel_origin)
        if discard:self.mels=self.mels[:,:,discard:].clone();self.mel_origin+=discard

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

    def warmup(self):
        started=time.perf_counter()
        # Grouped decoding must reach its first real Vocos call during startup,
        # rather than compiling that path on the first spoken sentence.
        count=4 if self.vocoder_batch_frames==1 and self.vc_group_chunks==1 else (self.algorithmic_buffer_ms+159)//160+self.vc_group_chunks+1
        for _ in range(count):self.process_chunk(np.zeros(2560,dtype=np.float32))
        self.stats['warmup_seconds']=time.perf_counter()-started;self.reset()

    def unload(self):
        self.vc=self.asr=self.vocos=None;self.kv=None

    def get_stats(self):return dict(self.stats,asr_position_rolls=self.position_rolls,asr_position_base=self.position_base)
