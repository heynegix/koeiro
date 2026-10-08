"""Checkpoint-free feature-grid tests; run in isolated MeanVC2 environment."""
import unittest
import hashlib
import io
from pathlib import Path
from unittest.mock import patch
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Run quality math tests in vc_models/meanvc2/.venv')
from src.vc.meanvc2 import MeanVC2Backend


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


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
