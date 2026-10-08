"""Calibrate this machine: GEMM throughput and per-op parallelism efficiency.

Establishes whether the streaming backend is compute-bound or dominated by
per-operation overhead and thread-scaling loss, before choosing an optimisation
strategy. Offline only; no audio devices.
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def timed(fn, count=5, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(count):
        fn()
    return (time.perf_counter() - t0) / count


def main():
    out = {'torch': torch.__version__,
           'cpu': os.environ.get('PROCESSOR_IDENTIFIER', ''),
           'logical_cores': os.cpu_count()}
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        # Large square GEMM: compute-bound reference.
        a = torch.randn(512, 512)
        b = torch.randn(512, 512)
        s = timed(lambda: a @ b, count=3, warmup=1)
        gf = 2 * 512 ** 3 / s / 1e9
        # Conv1d reference.
        x = torch.randn(1, 256, 19)
        w = torch.randn(256, 256, 3)
        sc = timed(lambda: torch.nn.functional.conv1d(x, w, padding=1))
        # Small GEMM: the regime the streaming DiT/Conformer actually runs in.
        small = torch.randn(16, 512)
        sb = torch.randn(512, 512)
        ss = timed(lambda: small @ sb, count=50)
        out[f'threads{threads}'] = {
            'gemm512_gflops': round(gf, 1),
            'conv1d_256x19_ms': round(sc * 1000, 3),
            'small_gemm_16x512x512_ms': round(ss * 1000, 4),
            'small_gemm_gflops': round(2 * 16 * 512 * 512 / ss / 1e9, 1),
        }
    report = ROOT / 'validation/v011/rtfopt/machine_calibration.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()