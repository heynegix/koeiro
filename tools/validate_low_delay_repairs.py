"""Offline gate audit on existing WAVs; no audio devices or model inference."""
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.processors.noise_gate import NoiseGateProcessor


def main():
    report = {'kind': 'OFFLINE WAV gate replay; no human listening or hardware latency measurement',
              'added_buffer_frames': 0, 'files': {}}
    paths = list((ROOT/'recordings/v011_mega_tournament/source').glob('source_*.wav'))
    paths.append(ROOT/'recordings/v011_meanvc2_continuity/source/source_long.wav')
    for path in paths:
        with wave.open(str(path), 'rb') as f:
            assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (48000, 1, 2)
            audio = np.frombuffer(f.readframes(f.getnframes()), '<i2').astype(np.float32)/32768
        metrics = {}
        for protected in (False, True):
            gate = NoiseGateProcessor(-45)
            gate.prepare(48000, 256)
            gate.protect_quiet = protected
            retained = quiet_energy = quiet_output = 0.
            elapsed = []
            for offset in range(0, len(audio), 256):
                original = audio[offset:offset+256]
                x = original.copy().reshape(-1, 1)
                start = time.perf_counter_ns()
                gate.process(x, 48000)
                elapsed.append((time.perf_counter_ns()-start)/1000)
                retained += float(np.sum(x*x))
                rms = float(np.sqrt(np.mean(original*original)))
                if 10**(-57/20) <= rms < 10**(-45/20):
                    quiet_energy += float(np.sum(original*original))
                    quiet_output += float(np.sum(x*x))
            metrics['protected' if protected else 'previous'] = {
                'quiet_band_energy_retention': quiet_output/quiet_energy if quiet_energy else None,
                'total_energy_retention': retained/float(np.sum(audio*audio)),
                'gate_p99_us': float(np.percentile(elapsed, 99))}
        report['files'][path.name] = metrics
    target = ROOT/'validation/v011/low_delay_gate_audit.json'
    target.write_text(json.dumps(report, indent=2), 'utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
