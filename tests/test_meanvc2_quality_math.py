"""Checkpoint-free feature-grid tests; run in isolated MeanVC2 environment."""
import unittest
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Run quality math tests in vc_models/meanvc2/.venv')
from src.vc.meanvc2 import MeanVC2Backend, retarget_devices


class _PinnedTrace(torch.nn.Module):
    """A trace that writes its capture device into the graph, like the shipped ASR."""

    def forward(self, x):
        return x.to(device=torch.device('cpu')) + 1.0


class _CudaPinnedTrace(torch.nn.Module):
    """A trace from a machine whose device this run cannot use, like issue #6."""

    def forward(self, x):
        return x.to(device=torch.device('cuda:63')) + 1.0


class FeatureGridTests(unittest.TestCase):
    def test_checkpoint_hashing_never_requests_unbounded_read(self):
        from tools.improve_meanvc2_quality import sha
        payload=b'checkpoint'*250000
        class BoundedStream(io.BytesIO):
            def read(self,size=-1):
                if not 0<size<=1024*1024:raise AssertionError('Unbounded checkpoint allocation')
                return super().read(size)
        with patch.object(Path,'open',lambda *args,**kwargs:BoundedStream(payload)):
            self.assertEqual(sha('unused'),hashlib.sha256(payload).hexdigest())

    def backend(self,mode):
        b=MeanVC2Backend(1);b.torch=torch;b.bn_interpolation=mode
        b.previous_bn=None;b.cond=torch.empty(1,0,256);b.noise=torch.empty(1,0,80)
        b.generator=torch.Generator().manual_seed(110)
        b.interpolation_weights=torch.arange(4)[None,None,:,None]/4
        return b

    def test_fixed_grid_continues_across_chunk_boundary(self):
        b=self.backend('fixed_linear')
        for i in range(3):b._append_bn(torch.arange(i*4,i*4+4).float()[None,:,None].expand(1,4,256))
        self.assertTrue(torch.equal(b.cond[0,16:,0],torch.arange(3,11,.25)))
        self.assertEqual(b.noise.shape,(1,48,80))

    def test_fixed_grid_constant_speaker_features_stay_constant(self):
        b=self.backend('fixed_linear')
        for _ in range(3):b._append_bn(torch.full((1,4,256),7.25))
        self.assertTrue(torch.equal(b.cond,torch.full((1,48,256),7.25)))

    def test_legacy_interpolation_remains_identical(self):
        b=self.backend('legacy');torch.manual_seed(19)
        frames=[torch.randn(1,4,256) for _ in range(3)];expected=[];previous=None
        for frame in frames:
            history=frame if previous is None else torch.cat((previous,frame),1)
            result=torch.nn.functional.interpolate(history.transpose(1,2),size=history.shape[1]*4,mode='linear',align_corners=True).transpose(1,2)
            expected.append(result if previous is None else result[:,4:]);previous=frame[:,-1:]
            b._append_bn(frame)
        self.assertTrue(torch.equal(b.cond,torch.cat(expected,1)))

    def test_aligned_frontend_keeps_correct_next_frame_origin(self):
        import torchaudio.compliance.kaldi as kaldi
        b=self.backend('legacy');b.kaldi=kaldi;b.feature_frontend='aligned';b.wave_tail=np.zeros(0,dtype=np.float32)
        wave=np.random.default_rng(8).normal(0,.05,2560*4).astype(np.float32)
        expected=kaldi.fbank(torch.from_numpy(wave*32768)[None],frame_length=25,frame_shift=10,snip_edges=True,num_mel_bins=80,energy_floor=0.,dither=0.,sample_frequency=16000)
        actual=torch.cat([b._extract_streaming_fbank(wave[i:i+2560]) for i in range(0,len(wave),2560)])
        self.assertEqual(actual.shape,expected.shape)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-5,rtol=1e-5))
        self.assertEqual(len(b.wave_tail),320)

    def loaded(self,device):
        b=MeanVC2Backend(1);b.torch=torch;b.feature_frontend='legacy';b.bn_interpolation='legacy'
        b.device=torch.device(device);b.vc=object();b.position_base=0
        return b

    def test_solver_noise_is_drawn_on_cpu_even_when_the_compute_device_is_not(self):
        """The seed has to mean the same thing on both devices.

        torch's CUDA generator is a different algorithm from the CPU one, so a
        device-local draw would hand a GPU run numbers the listened-to CPU route
        never produced, and the two routes could never be subtracted. 'meta' stands
        in for a compute device this machine does not have, and torch refuses to
        build a generator there, so a device-local stream cannot slip past.
        """
        b=self.loaded('meta');b.reset()
        self.assertEqual(b.generator.device.type,'cpu')
        draws=[];real=torch.randn
        def randn(*args,**kwargs):
            draws.append(kwargs.get('device'))
            return real(*args,**kwargs)
        b.torch=SimpleNamespace(randn=randn,cat=torch.cat,nn=torch.nn)
        with torch.inference_mode():
            cond,noise=b._condition_frames(torch.zeros(1,4,256))
        self.assertEqual(draws,[None],f'noise drawn for {draws[0]}, not the CPU stream')
        self.assertEqual(noise.device.type,'meta')
        self.assertEqual(cond.device.type,'cpu')


class RetargetDeviceTests(unittest.TestCase):
    def test_a_named_device_moves_and_the_matching_one_does_not(self):
        """The pass has to reach the graph that runs, not just the graph that prints.

        issue #6: on a GPU the ASR's position table stayed where the trace was
        captured and the add after it failed. Rewriting the device the trace names
        is what fixes it, and on the route the traces were captured on nothing moves
        at all, so the measured CPU run cannot be affected by it.
        """
        module=torch.jit.script(_PinnedTrace()).eval()
        self.assertEqual(retarget_devices([module.graph],torch.device('cpu')),0)
        self.assertEqual(retarget_devices([module.graph],torch.device('meta')),1)
        self.assertEqual(str(module(torch.ones(3)).device),'meta')

    def test_a_frozen_trace_follows_the_retarget_before_its_first_call(self):
        """The backend retargets inside load(); the first call happens in warmup.

        Freezing is what load() does, and a graph that has already run keeps the
        device it first executed with, so the order matters: rewrite, then call.
        This pins that order and proves a post-freeze rewrite reaches execution.
        """
        module=torch.jit.freeze(torch.jit.script(_PinnedTrace()).eval())
        self.assertEqual(retarget_devices([module.graph],torch.device('meta')),1)
        self.assertEqual(str(module(torch.ones(3)).device),'meta')

    def test_a_trace_pinned_to_an_unusable_device_runs_once_it_is_retargeted(self):
        """issue #6's direction: the graph names a device this run cannot use.

        The capture machine's device was baked into the ASR trace, so the op that
        touched the position table failed with 'found at least two devices'. cuda:63
        stands for any device this machine lacks, so the failure reproduces on CPU
        and CUDA builds alike; a fresh instance of the same trace runs on CPU once
        the constant is rewritten.
        """
        unfixed=torch.jit.freeze(torch.jit.script(_CudaPinnedTrace()).eval())
        with self.assertRaises(RuntimeError):
            unfixed(torch.ones(3))
        module=torch.jit.freeze(torch.jit.script(_CudaPinnedTrace()).eval())
        self.assertEqual(retarget_devices([module.graph],torch.device('cpu')),1)
        self.assertEqual(str(module(torch.ones(3)).device),'cpu')


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
