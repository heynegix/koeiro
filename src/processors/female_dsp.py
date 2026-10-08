from dataclasses import replace

from .base import AudioProcessor
from .dsp_parameters import DSPParameters
from .native_backend import NativeDSP


class FemaleDSPProcessor(AudioProcessor):
    """Female DSP / zero-delay Original, implemented outside Audio I/O.

    Publish one immutable parameter object from the UI. Native state and buffers
    belong exclusively to the callback; nothing is loaded on a parameter change.
    Algorithm quality is chosen during prepare, so change that while stopped.
    """

    def __init__(self, parameters=None):
        self.parameters = parameters or DSPParameters()
        self._native = None
        self._prepared_quality = None

    def set_parameters(self, parameters):
        if not isinstance(parameters, DSPParameters):
            raise TypeError("Expected validated DSPParameters")
        self.parameters = parameters

    def update(self, **changes):
        self.set_parameters(replace(self.parameters, **changes))

    @property
    def algorithmic_latency_samples(self):
        return self._native.latency_samples if self._native and self.parameters.mode == "female_dsp" else 0

    def prepare(self, sample_rate, max_frames):
        if self._native:
            self._native.close()
        self._native = NativeDSP(sample_rate, max_frames, self.parameters.quality)
        self._prepared_quality = self.parameters.quality
        self._sample_rate = sample_rate

    def reset(self):
        if self._native:
            self._native.reset()

    def process(self, audio, sample_rate):
        if not self._native:
            raise RuntimeError("Female DSP must be prepared before processing")
        if sample_rate != self._sample_rate or self.parameters.quality != self._prepared_quality:
            raise RuntimeError("Stop and restart before changing DSP quality or sample rate")
        return self._native.process(audio, self.parameters)
