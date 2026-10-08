import math

import numpy as np


def peak_level(audio: np.ndarray) -> float:
    """Peak amplitude without allocating an absolute-value sample buffer."""
    return max(abs(float(np.min(audio))), abs(float(np.max(audio))))


def amplitude_to_db(level: float, floor: float = -80.0) -> float:
    if not math.isfinite(level) or level <= 0:
        return floor
    return max(floor, 20.0 * math.log10(level))
