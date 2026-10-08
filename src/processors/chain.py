import numpy as np
import time

from src.audio.performance import TimingStats
from src.vc.models import is_meanvc2

from .base import AudioProcessor
from .gain import GainProcessor
from .noise_gate import NoiseGateProcessor
from .passthrough import PassthroughProcessor


class SafetyClampProcessor(AudioProcessor):
    def prepare(self, sample_rate, max_frames):
        self._finite = np.empty((max_frames, 1), dtype=bool)

    def process(self, audio, sample_rate):
        # Reuse a boolean mask rather than nan_to_num's temporary masks.
        mask = self._finite[:len(audio)]
        np.isfinite(audio, out=mask)
        np.logical_not(mask, out=mask)
        np.copyto(audio, 0.0, where=mask)
        np.clip(audio, -1.0, 1.0, out=audio)
        return audio


class ProcessorChain(AudioProcessor):
    def __init__(self, main_processor=None, gain_db=0.0, threshold_db=-45.0):
        self.gate = NoiseGateProcessor(threshold_db)
        self.gain = GainProcessor(gain_db)
        self.pre_processors = (self.gate, self.gain)
        self.main_processor = main_processor if main_processor is not None else PassthroughProcessor()
        self.post_processors = (SafetyClampProcessor(),)
        self._stages = (*self.pre_processors, self.main_processor, *self.post_processors)
        self.performance = TimingStats()

    def prepare(self, sample_rate, max_frames):
        self.performance.reset()
        for processor in self._stages:
            processor.prepare(sample_rate, max_frames)

    def reset(self):
        for processor in self._stages:
            processor.reset()

    def stop(self):
        for processor in self._stages:
            processor.stop()

    def process(self, audio, sample_rate):
        shape = audio.shape
        router = self.main_processor
        ai = getattr(router, 'ai', None)
        self.gate.protect_quiet = (getattr(router, 'mode', None) == 'ai_voice'
                                  and ai is not None and is_meanvc2(ai.bridge.parameters.model))
        for processor in self._stages:
            started = time.perf_counter_ns() if processor is self.main_processor else 0
            audio = processor.process(audio, sample_rate)
            if started:
                self.performance.record(time.perf_counter_ns() - started, len(audio), sample_rate)
            if not isinstance(audio, np.ndarray) or audio.shape != shape or audio.dtype != np.float32:
                raise ValueError("Processor must return float32 mono with unchanged frame count")
        return audio
