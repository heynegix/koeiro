import math

import numpy as np

from .base import AudioProcessor


class NoiseGateProcessor(AudioProcessor):
    """Block RMS detector, 3 dB hysteresis, 50 ms hold, 5/80 ms attack/release."""

    def __init__(self, threshold_db=-45.0):
        self.set_threshold(threshold_db)
        self.protect_quiet = False
        self.reset()

    def set_threshold(self, threshold_db):
        if not math.isfinite(threshold_db) or not -80 <= threshold_db <= -10:
            raise ValueError("Noise gate must be between -80 and -10 dB")
        # Publish both thresholds together, for consistent callback reads.
        self._thresholds = (10.0 ** (threshold_db / 20),
                            10.0 ** ((threshold_db - 3) / 20))

    def prepare(self, sample_rate, max_frames):
        times = np.arange(1, max_frames + 1, dtype=np.float32) / sample_rate
        self._attack = np.exp(-times / 0.005).reshape(-1, 1)
        self._speech_attack = np.exp(-times / 0.001).reshape(-1, 1)
        self._release = np.exp(-times / 0.080).reshape(-1, 1)
        self._envelope = np.empty((max_frames, 1), dtype=np.float32)
        self._square = np.empty((max_frames, 1), dtype=np.float32)
        self._hold_frames = round(sample_rate * 0.050)
        self._speech_hold_frames = round(sample_rate * 0.100)
        self.reset()

    def reset(self):
        self._level = 0.0
        self._opened = False
        self._hold = 0

    def process(self, audio, sample_rate):
        frames = len(audio)
        square = self._square[:frames]
        np.multiply(audio, audio, out=square)
        rms = math.sqrt(float(np.mean(square)))
        open_threshold, close_threshold = self._thresholds
        if rms >= (close_threshold if self._opened else open_threshold):
            self._opened = True
            self._hold = self._speech_hold_frames if self.protect_quiet else self._hold_frames
        elif self._hold > 0:
            self._hold = max(0, self._hold - frames)
        else:
            self._opened = False
        target = 1.0 if self._opened else 0.0
        if self.protect_quiet and not self._opened and rms >= open_threshold * .25:
            # Soft knee retains low consonants, without buffering their onset.
            target = min(1.0, math.sqrt(rms / open_threshold))
        envelope = self._envelope[:frames]
        attack = self._speech_attack if self.protect_quiet else self._attack
        decay = attack if target > self._level else self._release
        np.multiply(decay[:frames], self._level - target, out=envelope)
        np.add(envelope, target, out=envelope)
        self._level = float(envelope[-1, 0])
        np.multiply(audio, envelope, out=audio)
        return audio
