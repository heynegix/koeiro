from dataclasses import dataclass
import math

from .formant import validate_formant
from .pitch import validate_pitch

MODES = ("original", "female_dsp")
QUALITIES = ("low_latency", "balanced")


@dataclass(frozen=True)
class DSPParameters:
    mode: str = "original"
    pitch: float = 3.0
    formant: float = 2.0
    brightness: float = 60.0
    low_cut: bool = True
    limiter: bool = True
    wet: float = 1.0
    quality: str = "low_latency"

    def __post_init__(self):
        if self.mode not in MODES or self.quality not in QUALITIES:
            raise ValueError("Invalid DSP mode or quality")
        validate_pitch(self.pitch)
        validate_formant(self.formant)
        for name, low, high in (("brightness", 0, 100), ("wet", 0, 1)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
                raise ValueError(f"Invalid {name}")
        if type(self.low_cut) is not bool or type(self.limiter) is not bool:
            raise ValueError("Low Cut and Limiter must be boolean")
