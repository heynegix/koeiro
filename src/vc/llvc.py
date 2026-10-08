import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .base import VoiceConversionBackend


class LLVCBackend(VoiceConversionBackend):
    """CPU cached LLVC. Imported Torch and model stay in the isolated AI process."""
    def __init__(self, chunk_factor=1, threads=1):
        if type(chunk_factor) is not int or chunk_factor not in (1, 2, 4):
            raise ValueError('LLVC chunk factor must be 1, 2 or 4')
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError('CPU threads must be 1..4')
        self.chunk_factor, self.threads = chunk_factor, threads
        self.chunk_samples = 208*chunk_factor
        self.model = None
        self.stats = {}

    def load(self, model_path):
        start = time.perf_counter()
        folder = Path(model_path)
        metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        checkpoint = folder/'model.pth'
        if checkpoint.stat().st_size > 100_000_000:
            raise ValueError('Unexpected LLVC model size')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError('LLVC model checksum mismatch')
        if metadata.get('sample_rate') != self.sample_rate:
            raise ValueError('Unsupported model sample rate')
        import torch
        from third_party.llvc.model import Net
        self.torch = torch
        torch.set_num_threads(self.threads)
        # set_num_interop_threads may only be called before the first inference.
        if not getattr(LLVCBackend, '_interop_set', False):
            torch.set_num_interop_threads(1)
            LLVCBackend._interop_set = True
        config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
        self.model = Net(**config['model_params']).eval()
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights['model'], strict=True)
        self.reset()
        self.stats = dict(load_seconds=time.perf_counter()-start, model_bytes=checkpoint.stat().st_size,
                          torch=torch.__version__, threads=self.threads, sample_rate=self.sample_rate,
                          chunk_samples=self.chunk_samples, model_alignment_delay_ms=1.0)

    def reset(self):
        if self.model is None:
            return
        self.enc, self.dec, self.out = self.model.init_buffers(1, 'cpu')
        self.prenet = self.model.convnet_pre.init_ctx_buf(1, 'cpu')
        self.previous = np.zeros(2*self.model.L, dtype=np.float32)

    def warmup(self):
        start = time.perf_counter()
        silence = np.zeros(self.chunk_samples, dtype=np.float32)
        for _ in range(12):
            self.process_chunk(silence)
        self.stats['warmup_seconds'] = time.perf_counter()-start
        self.reset()

    def process_chunk(self, audio):
        if self.model is None:
            raise RuntimeError('LLVC is not loaded')
        if audio.dtype != np.float32 or audio.ndim != 1 or len(audio) != self.chunk_samples:
            raise ValueError('LLVC expects one exact float32 mono chunk')
        if not np.isfinite(audio).all():
            raise ValueError('Non-finite LLVC input')
        combined = np.concatenate((self.previous, audio))
        self.previous[:] = audio[-len(self.previous):]
        with self.torch.inference_mode():
            result, self.enc, self.dec, self.out, self.prenet = self.model(
                self.torch.from_numpy(combined).reshape(1, 1, -1),
                self.enc, self.dec, self.out, self.prenet, pad=False)
        converted = result.reshape(-1).numpy().copy()
        if len(converted) != len(audio) or not np.isfinite(converted).all():
            raise RuntimeError('Invalid LLVC output')
        return np.clip(converted, -1, 1)

    def unload(self):
        self.model = None
        self.enc = self.dec = self.out = self.prenet = None
        self.previous = None

    def get_stats(self):
        return dict(self.stats)
