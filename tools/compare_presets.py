"""Offline comparison WAVs from one PCM16 recording or a labelled synthetic vowel.

This utility never writes a file from an audio callback. --capture is opt-in
physical-microphone capture to memory first; processing/writing happens later.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.performance import TimingStats
from src.presets.female_presets import PRESETS, load_preset
from src.processors.female_dsp import FemaleDSPProcessor


def read_wav(path):
    with wave.open(str(path), "rb") as source:
        rate, channels = source.getframerate(), source.getnchannels()
        if rate not in (44100, 48000) or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError("Use an uncompressed PCM16 WAV at 44100/48000 Hz")
        if source.getnframes() > rate*600:
            raise ValueError("Comparison input must be 10 minutes or shorter")
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32)
    return audio.reshape(-1, channels).mean(axis=1, dtype=np.float32)/32768, rate


def write_wav(path, audio, rate):
    pcm = np.round(np.clip(audio, -1, 1)*32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def synthetic_vowel(rate=48000):
    # Analytical harmonic signal with changing vowel resonances and F0. This
    # is explicitly NOT a human speech recording or a perceptual voice test.
    parts = []
    for f0, f1, f2 in ((120, 720, 1440), (135, 400, 2000), (110, 300, 1000), (145, 550, 1700)):
        t = np.arange(rate*2)/rate
        audio = np.zeros(len(t))
        for harmonic in range(1, 75):
            f = harmonic*f0
            amplitude = (0.015/harmonic + np.exp(-0.5*((f-f1)/120)**2)
                         + 0.5*np.exp(-0.5*((f-f2)/180)**2))
            audio += amplitude*np.sin(2*np.pi*f*t)
        envelope = np.minimum(1, t/0.04)*np.minimum(1, (2-t)/0.08)
        parts.append((audio/max(abs(audio))*0.25*envelope).astype(np.float32))
    return np.concatenate(parts), rate


def render(audio, rate, parameters, block=256):
    processor = FemaleDSPProcessor(parameters)
    processor.prepare(rate, block)
    delay = processor.algorithmic_latency_samples
    # Warm up enable/crossfade before the recording, then compensate signal
    # delay in the saved files. Live output still has this physical delay.
    for _ in range((delay+rate//10)//block+1):
        processor.process(np.zeros((block, 1), dtype=np.float32), rate)
    padded = np.pad(audio, (0, delay))
    output = np.empty_like(padded)
    timing = TimingStats()
    for offset in range(0, len(padded), block):
        chunk = padded[offset:offset+block].reshape(-1, 1).copy()
        start = time.perf_counter_ns()
        processor.process(chunk, rate)
        timing.record(time.perf_counter_ns()-start, len(chunk), rate)
        output[offset:offset+len(chunk)] = chunk[:, 0]
    return output[delay:delay+len(audio)], delay, timing.snapshot()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--capture", type=int, metavar="DEVICE", help="Opt-in physical microphone index")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--quality", choices=["low_latency", "balanced"], default="low_latency")
    args = parser.parse_args()
    if args.capture is not None:
        import sounddevice as sd
        if not 1 <= args.seconds <= 600:
            parser.error("seconds must be 1..600")
        device = sd.query_devices(args.capture)
        channels = min(2, device["max_input_channels"])
        if channels < 1 or "cable" in device["name"].lower() or "voicemeeter" in device["name"].lower():
            parser.error("Choose a physical microphone")
        host = sd.query_hostapis(device["hostapi"])["name"]
        extra = sd.WasapiSettings(auto_convert=True) if "wasapi" in host.lower() else None
        print(f"Recording {args.seconds:g}s. Read the same test sentences now.", flush=True)
        rate = 48000
        recorded = sd.rec(round(args.seconds*rate), samplerate=rate, channels=channels,
                          dtype="float32", device=args.capture, extra_settings=extra, blocking=True)
        audio = recorded.mean(axis=1, dtype=np.float32)
        description = f"Physical microphone: {device['name']}"
    elif args.synthetic:
        audio, rate = synthetic_vowel()
        description = "Synthetic harmonic vowel fixture; NOT human speech"
    else:
        audio, rate = read_wav(args.input)
        description = str(args.input.resolve())
    if not len(audio):
        parser.error("Input is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"source": description, "sample_rate": rate, "seconds": len(audio)/rate,
              "alignment": "Saved DSP files compensate algorithmic delay; live output does not", "presets": {}}
    for name in PRESETS:
        parameters = load_preset(name, args.quality)
        output, delay, timing = render(audio, rate, parameters)
        path = args.output_dir / (name.replace(" ", "_")+".wav")
        write_wav(path, output, rate)
        report["presets"][name] = {"parameters": asdict(parameters), "algorithmic_latency_ms": delay/rate*1000,
                                    "offline_processing": asdict(timing), "peak": float(np.max(np.abs(output)))}
        print(path, flush=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
