import hashlib
import json
from pathlib import Path
import time

import numpy as np
from .base import VoiceConversionBackend


class LLVCOnnxBackend(VoiceConversionBackend):
    """Equivalent LLVC graph; CPU-only ONNX runtime in the AI subprocess."""
    def __init__(self, chunk_factor=1, threads=1):
        if type(chunk_factor) is not int or chunk_factor not in (1, 2, 4):
            raise ValueError('LLVC chunk factor must be 1, 2 or 4')
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError('CPU threads must be 1..4')
        self.chunk_samples, self.factor, self.threads = 208*chunk_factor, chunk_factor, threads
        self.session = None
        self.stats = {}

    def load(self, model_path):
        start = time.perf_counter()
        folder = Path(model_path)
        manifest = json.loads((folder/'onnx.json').read_text(encoding='utf-8'))
        model = manifest['graphs'][str(self.factor)]
        # File names come from the local installer/exporter; confine them too.
        path = folder/model['filename']
        if path.resolve().parent != folder.resolve():
            raise ValueError('Invalid model path')
        if path.stat().st_size > 100_000_000 or hashlib.sha256(path.read_bytes()).hexdigest() != model['sha256']:
            raise ValueError('LLVC ONNX checksum/size mismatch')
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.names = [item.name for item in self.session.get_inputs()]
        self.reset()
        self.stats = dict(load_seconds=time.perf_counter()-start, model_bytes=path.stat().st_size,
                          onnxruntime=ort.__version__, threads=self.threads, sample_rate=16000,
                          chunk_samples=self.chunk_samples, model_alignment_delay_ms=1.0,
                          backend='LLVC / ONNX CPU', graph_sha256=model['sha256'])

    def reset(self):
        if self.session is None:
            return
        self.state = [np.zeros(item.shape, dtype=np.float32) for item in self.session.get_inputs()[1:]]
        self.previous = np.zeros(32, dtype=np.float32)

    def warmup(self):
        start = time.perf_counter()
        for _ in range(12):
            self.process_chunk(np.zeros(self.chunk_samples, dtype=np.float32))
        self.stats['warmup_seconds'] = time.perf_counter()-start
        self.reset()

    def process_chunk(self, audio):
        if self.session is None:
            raise RuntimeError('LLVC is not loaded')
        if audio.ndim != 1 or audio.dtype != np.float32 or len(audio) != self.chunk_samples or not np.isfinite(audio).all():
            raise ValueError('LLVC expects one finite float32 mono chunk')
        inputs = [np.concatenate((self.previous, audio)).reshape(1, 1, -1), *self.state]
        output, *self.state = self.session.run(None, dict(zip(self.names, inputs)))
        self.previous[:] = audio[-32:]
        result = output.reshape(-1)
        if len(result) != len(audio) or not np.isfinite(result).all():
            raise RuntimeError('Invalid LLVC output')
        return np.clip(result, -1, 1)

    def unload(self):
        self.session = None
        self.state = self.previous = None

    def get_stats(self):
        return dict(self.stats)
