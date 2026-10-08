import math


def validate_pitch(value):
    if type(value) not in (int, float) or not -12 <= value <= 12 or not math.isfinite(value):
        raise ValueError("Pitch must be between -12 and +12 semitones")
    return float(value)


def pitch_factor(semitones):
    return 2 ** (validate_pitch(semitones) / 12)
