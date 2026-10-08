"""Opt-in real-device check. Only statistics are saved, never microphone samples."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sounddevice as sd

from src.audio.devices import enumerate_devices
from src.audio.controller import AudioController
from src.audio.engine import AudioEngine, EngineConfig
from src.audio.meters import peak_level
from src.processors.chain import ProcessorChain
from src.processors.female_dsp import FemaleDSPProcessor
from src.processors.base import AudioProcessor
from src.processors.passthrough import PassthroughProcessor
from src.presets.female_presets import PRESETS, load_preset
from src.utils.logging import configure_logging


class VerificationSignal(AudioProcessor):
    """Developer-only controlled input at the Main slot, outside the engine."""
    def __init__(self, processor, wav=None):
        from tools.compare_presets import synthetic_vowel, read_wav
        signal, rate = read_wav(wav) if wav else synthetic_vowel()
        if rate != 48000:
            raise ValueError("Verification WAV must be 48000 Hz")
        self.signal = signal.reshape(-1, 1)
        self.processor = processor
        self.offset = 0

    def prepare(self, rate, frames):
        if rate != 48000:
            raise ValueError("Synthetic verification uses 48000 Hz")
        self.offset = 0
        self.processor.prepare(rate, frames)

    def process(self, audio, rate):
        count = min(len(audio), len(self.signal)-self.offset)
        audio[:count] = self.signal[self.offset:self.offset+count]
        if count < len(audio):
            audio[count:] = self.signal[:len(audio)-count]
        self.offset = (self.offset+len(audio)) % len(self.signal)
        return self.processor.process(audio, rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=int, required=True, help="Physical microphone PortAudio index")
    parser.add_argument("--output", type=int, required=True, help="CABLE Input PortAudio index")
    parser.add_argument("--cable-return", type=int, help="Optional CABLE Output capture index")
    parser.add_argument("--buffer", type=int, default=256, choices=[64, 128, 256, 512, 1024])
    parser.add_argument("--sample-rate", type=int, default=48000, choices=[44100, 48000])
    parser.add_argument("--seconds", type=float, default=5, help="Seconds per start/stop cycle")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--gain", type=float, default=0.0, help="Gain dB (-20 to +20)")
    parser.add_argument("--gate", type=float, default=-45.0, help="Gate dB (-80 to -10)")
    parser.add_argument("--report", type=Path, default=Path("test-results/hardware.json"))
    parser.add_argument("--preset", choices=list(PRESETS), help="Omit for v0.1 Passthrough baseline")
    parser.add_argument("--quality", choices=["low_latency", "balanced"], default="low_latency")
    parser.add_argument("--exercise-parameters", action="store_true", help="Cycle presets and sliders without restarting")
    parser.add_argument("--test-signal", action="store_true", help="Replace mic at Main slot with a labelled synthetic vowel")
    parser.add_argument("--test-wav", type=Path, help="Replay an existing human WAV at Main; never record the microphone")
    parser.add_argument("--speech-presets", action="store_true", help="Cycle only the three enabled DSP presets, every 10 seconds")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.cycles < 1:
        parser.error("seconds and cycles must be positive")
    if args.seconds*args.cycles>840:
        parser.error('Hardware trial must fit within 15 minutes including cleanup')
    configure_logging(Path("test-results/logs"))
    devices = {d.index: d for d in enumerate_devices()}
    input_device, output_device = devices[args.input], devices[args.output]
    if "cable input" not in output_device.name.lower():
        parser.error("This check only outputs to CABLE Input; select its listed index")
    reader = None
    return_peak = 0.0
    return_callbacks = 0
    return_xruns = 0
    def receive(audio, frames, timing, status):
        nonlocal return_peak, return_callbacks, return_xruns
        return_peak = max(return_peak, peak_level(audio))
        return_callbacks += 1
        if status:
            return_xruns += 1
    dsp = FemaleDSPProcessor(load_preset(args.preset, args.quality)) if args.preset else None
    main_processor = dsp
    if args.test_signal or args.test_wav:
        main_processor = VerificationSignal(dsp or PassthroughProcessor(), args.test_wav)
    sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
    engine = AudioEngine(ProcessorChain(main_processor=main_processor))
    engine.chain.gain.set_gain(args.gain)
    engine.chain.gate.set_threshold(args.gate)
    controller = None
    report = {"sample_rate": args.sample_rate, "buffer_size": args.buffer,
              "input": input_device.label, "output": output_device.label,
              "seconds_per_cycle": args.seconds, "cycles": [], "error": None}
    report.update(gain_db=args.gain, gate_db=args.gate, control_thread=True)
    report.update(preset=args.preset or "v0.1 Passthrough", quality=args.quality)
    report["main_input"] = "Synthetic harmonic vowel (not microphone speech)" if args.test_signal else "Physical microphone"
    if args.test_wav:
        report["main_input"] = "Human WAV replay at Main (not live conversation): " + str(args.test_wav)

    def wait_request(request, state):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            snapshot = controller.snapshot
            if snapshot.request_id == request:
                if snapshot.state in ("Error", "Restart required"):
                    raise RuntimeError(snapshot.error)
                if snapshot.state == state:
                    return
            time.sleep(0.01)
        raise RuntimeError("Audio control thread did not finish the request")
    started = time.monotonic()
    try:
        if args.cable_return is not None:
            cable = devices[args.cable_return]
            if "cable output" not in cable.name.lower():
                parser.error("cable-return must be CABLE Output")
            extra = sd.WasapiSettings(auto_convert=True) if "wasapi" in cable.host_api.lower() else None
            reader = sd.InputStream(device=cable.index, channels=min(2, cable.max_input_channels),
                                    dtype="float32", samplerate=args.sample_rate,
                                    blocksize=args.buffer, extra_settings=extra, callback=receive)
            reader.start()
        for cycle in range(args.cycles):
            if controller is None:
                controller = AudioController(engine)
            request = controller.start(EngineConfig(input_device, output_device, args.sample_rate, args.buffer))
            wait_request(request, "Running")
            deadline = time.monotonic() + args.seconds
            cycle_start, cpu_start = time.monotonic(), time.process_time()
            loads = []
            changes = 0
            input_peak = output_peak = 0.0
            while time.monotonic() < deadline:
                if controller.snapshot.state != "Running":
                    raise RuntimeError(controller.snapshot.error or "Audio stream stopped")
                current_input, current_output = engine.take_peaks()
                input_peak = max(input_peak, current_input)
                output_peak = max(output_peak, current_output)
                loads.append(engine.cpu_load)
                if dsp and args.exercise_parameters:
                    dsp.set_parameters(load_preset(list(PRESETS)[changes % 4], args.quality))
                    dsp.update(pitch=changes % 241/10-12, formant=changes % 121/10-6,
                               brightness=changes % 101)
                    changes += 1
                if dsp and args.speech_presets:
                    desired = int((time.monotonic()-cycle_start)/10) % 3
                    if desired != changes % 3 or changes == 0:
                        dsp.set_parameters(load_preset(["Female Soft", "Female Bright", "Anime Test"][desired], args.quality))
                        engine.diagnostics.note_gui_action()
                        changes = desired if desired else 3
                time.sleep(0.02)
            record = {"cycle": cycle + 1, "input_peak": input_peak, "output_peak": output_peak,
                      "underflows": engine.underflows, "overflows": engine.overflows,
                      "reported_io_latency_ms": engine.reported_latency_ms,
                      "stream_channels": [engine.input_channels, engine.output_channels]}
            record.update(dsp=asdict(engine.chain.performance.snapshot()),
                          callback=asdict(engine.performance.snapshot()),
                          algorithmic_latency_ms=(dsp.algorithmic_latency_samples / args.sample_rate * 1000 if dsp else 0),
                          deadline_ms=args.buffer / args.sample_rate * 1000,
                          portaudio_cpu_load_average=sum(loads)/max(1, len(loads)),
                          portaudio_cpu_load_maximum=max(loads, default=0),
                          process_cpu_one_core_percent=100*(time.process_time()-cpu_start)/max(0.001, time.monotonic()-cycle_start),
                          parameter_changes=changes)
            record["diagnostics"] = engine.diagnostics.snapshot()
            if dsp:
                record["native_dsp"] = dsp._native.timing_snapshot()
            wait_request(controller.stop(), "Stopped")
            report["cycles"].append(record)
            print(json.dumps(record), flush=True)
    except Exception as error:
        report["error"] = str(error)
        print(f"Verification failed: {error}", file=sys.stderr)
    finally:
        try:
            if controller is not None:
                controller.shutdown()
                deadline = time.monotonic() + 10
                while controller.alive and time.monotonic() < deadline:
                    time.sleep(0.01)
                if controller.alive:
                    report["error"] = "Control thread did not shut down"
        finally:
            if reader is not None:
                reader.close()
        report.update(elapsed_seconds=time.monotonic() - started,
                      cable_return_peak=return_peak, cable_return_callbacks=return_callbacks,
                      cable_return_xruns=return_xruns)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if report["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
