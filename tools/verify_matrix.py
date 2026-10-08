"""Run the opt-in v0.2 hardware matrix; no microphone audio is saved."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=int, required=True)
    parser.add_argument("--output", type=int, required=True)
    parser.add_argument("--cable-return", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = [sys.executable, str(root / "tools/verify_audio.py"), "--input", str(args.input),
            "--output", str(args.output), "--cable-return", str(args.cable_return), "--gate", "-80"]
    runs = [("baseline", 256, None, "low_latency", 1, args.seconds, False)]
    runs += [(f"female-{block}", block, "Female Soft", "low_latency", 1, args.seconds, False)
             for block in (64, 128, 256, 512, 1024)]
    runs += [("balanced-256", 256, "Female Soft", "balanced", 1, args.seconds, False),
             ("live-switching", 256, "Female Soft", "low_latency", 10, 2, True)]
    for name, block, preset, quality, cycles, seconds, exercise in runs:
        print(f"Running {name}", flush=True)
        command = base + ["--buffer", str(block), "--cycles", str(cycles), "--seconds", str(seconds),
                          "--quality", quality, "--report", str(root / f"test-results/v02-{name}.json")]
        if preset:
            command += ["--preset", preset]
        if exercise:
            command += ["--exercise-parameters"]
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
