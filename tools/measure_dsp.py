"""Reproducible analytical Pitch/Formant measurements, without audio hardware."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compare_presets import render, synthetic_vowel
from src.processors.dsp_parameters import DSPParameters


def spectrum(audio, rate=48000):
    tail = audio[rate//2:]
    return np.fft.rfftfreq(len(tail), 1/rate), np.abs(np.fft.rfft(tail*np.hanning(len(tail))))**2


def centroid(audio):
    frequency, power = spectrum(audio)
    selected = (frequency >= 350) & (frequency <= 1150)
    return float(np.sum(frequency[selected]*power[selected])/np.sum(power[selected]))


def main():
    rate = 48000
    report = {"sample_rate": rate, "input": "Analytical tones / synthetic vowel, not human speech",
              "pitch": [], "formant": []}
    for quality in ("low_latency", "balanced"):
        base = DSPParameters(mode="female_dsp", quality=quality, brightness=50, low_cut=False, formant=0)
        for f0 in (120, 220, 440):
            source = (0.2*np.sin(2*np.pi*f0*np.arange(rate*2)/rate)).astype(np.float32)
            for semitones in (-12, -3, 0, 3, 12):
                output, _, _ = render(source, rate, replace(base, pitch=semitones))
                frequency, power = spectrum(output)
                detected = float(frequency[np.argmax(power)])
                target = f0*2**(semitones/12)
                report["pitch"].append(dict(quality=quality, input_hz=f0, semitones=semitones,
                    target_hz=target, output_hz=detected, error_cents=float(1200*np.log2(detected/target))))
        source = synthetic_vowel()[0][:rate*2]
        for pitch, formant in ((0, -4), (0, 0), (0, 4), (4, 0), (4, 4)):
            output, _, _ = render(source, rate, replace(base, pitch=pitch, formant=formant))
            report["formant"].append(dict(quality=quality, pitch=pitch, formant=formant,
                first_resonance_band_centroid_hz=centroid(output)))
    path = Path("test-results/v02-signals.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
