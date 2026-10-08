"""Worker-only causal FIR integer-ratio resampling with continuous filter state."""
import numpy as np


class StreamingResampler:
    def __init__(self, input_rate, output_rate):
        if (input_rate, output_rate) not in ((48000, 16000), (16000, 48000)):
            raise ValueError('Only 48k <-> 16k is supported')
        from scipy.signal import firwin, lfilter
        self._lfilter = lfilter
        self.down = input_rate == 48000
        self.kernel = firwin(97, 6800, fs=48000, window=('kaiser', 7)).astype(np.float32)
        if not self.down:
            self.kernel *= 3
        self.delay_ms = 1.0  # 48 high-rate samples, each direction.
        self.reset()

    def reset(self):
        self.state = np.zeros(len(self.kernel)-1, dtype=np.float32)
        self.input_count = 0

    def process(self, audio):
        if audio.ndim != 1 or audio.dtype != np.float32 or not np.isfinite(audio).all():
            raise ValueError('Resampler input must be finite float32 mono')
        if not len(audio):
            return np.empty(0, dtype=np.float32)
        if self.down:
            filtered, self.state = self._lfilter(self.kernel, [1.0], audio, zi=self.state)
            start = (-self.input_count) % 3
            self.input_count += len(audio)
            return filtered[start::3].astype(np.float32)
        padded = np.zeros(3*len(audio), dtype=np.float32)
        padded[::3] = audio
        filtered, self.state = self._lfilter(self.kernel, [1.0], padded, zi=self.state)
        return filtered.astype(np.float32)
