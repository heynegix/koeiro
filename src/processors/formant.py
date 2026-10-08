import math


def validate_formant(value):
    if type(value) not in (int, float) or not -6 <= value <= 6 or not math.isfinite(value):
        raise ValueError("Formant must be between -6 and +6 semitones")
    return float(value)


def formant_factor(semitones):
    return 2 ** (validate_formant(semitones) / 12)
