from dataclasses import replace

from src.processors.dsp_parameters import DSPParameters

PRESETS = {
    "Original": DSPParameters(),
    "Female Soft": DSPParameters(mode="female_dsp", pitch=3.0, formant=1.5, brightness=55),
    "Female Bright": DSPParameters(mode="female_dsp", pitch=4.5, formant=2.5, brightness=65),
    "Anime Test": DSPParameters(mode="female_dsp", pitch=6.0, formant=3.0, brightness=75),
}


def load_preset(name, quality="low_latency"):
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    return replace(PRESETS[name], quality=quality)
