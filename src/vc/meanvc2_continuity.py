"""Offline-only continuity experiments; never selected by the production app.

Reference, weights, feature frontend, interpolation and noise remain unchanged.
Larger inference groups retain the checkpoint's 120ms chunk attention mask.
They do not give the network an arbitrary new attention window.
"""
import math
import time

from .meanvc2 import MeanVC2Backend


class MeanVC2ContinuityBackend(MeanVC2Backend):
    def __init__(self, threads=4, vc_chunks=1, decode_frames=1, device='cpu'):
        if type(vc_chunks) is not int or vc_chunks not in (1, 3, 6):
            raise ValueError('VC groups must be 1, 3 or 6 chunks')
        if type(decode_frames) is not int or decode_frames not in (1, 36):
            raise ValueError('Decode group must be 1 or 36 frames')
        self.vc_chunks = vc_chunks
        self.decode_frames = decode_frames
        super().__init__(threads, device)

    @property
    def algorithmic_buffer_ms(self):
        # Conservative whole-input-hop buffering covers grouped production.
        extra_frames = (self.vc_chunks - 1) * 12 + self.decode_frames - 1
        base = 640 if self.feature_frontend == 'aligned' else 480
        return base + math.ceil(extra_frames / 16) * 160

    def load(self, model_path):
        super().load(model_path)
        # The explicit experiment condition overrides a production decode
        # grouping setting so the saved original control stays reproducible.
        self.vocoder_batch_frames = 1
        self.vc_group_chunks = 1
        self.reset()

    def _convert_blocks(self):
        if self.vc_chunks == 1:
            return super()._convert_blocks()
        t = self.torch
        emitted = self.vc_chunks * 12
        required = emitted + 4
        while self.cond.shape[1] >= required:
            cond = self.cond[:, :required]
            x = self.noise[:, :required].clone()
            offset = 0 if self.kv is None else self.kv[0][0].shape[2]
            for step in range(self.steps):
                u, cache = self.vc(
                    x, *self.timesteps[step], cache=None, cond=cond,
                    spks=self.speaker, offset=offset, is_inference=True,
                    kv_cache=self.kv)
                x = x - (1 / self.steps) * u
            self.kv = [(k[:, :, -self.max_kv_frames:].contiguous(),
                        v[:, :, -self.max_kv_frames:].contiguous())
                       for k, v in cache]
            self.mels = t.cat((self.mels, (x[:, :emitted].transpose(1, 2) + 1) / 2), 2)
            self.cond = self.cond[:, emitted:]
            self.noise = self.noise[:, emitted:]

    def _decode_ready(self):
        ready = self.mel_origin + self.mels.shape[2] - self.vocoder_right
        if ready - self.decoded_frames < self.decode_frames:
            return
        super()._decode_ready()

    def warmup(self):
        import numpy as np
        start = time.perf_counter()
        for _ in range(math.ceil(self.algorithmic_buffer_ms / 160) + self.vc_chunks + 4):
            self.process_chunk(np.zeros(2560, dtype=np.float32))
        self.stats['warmup_seconds'] = time.perf_counter() - start
        self.stats.update(experiment_only=True, vc_group_chunks=self.vc_chunks,
                          decode_group_frames=self.decode_frames,
                          algorithmic_buffer_ms=self.algorithmic_buffer_ms,
                          human_quality_accepted=False)
        self.reset()
