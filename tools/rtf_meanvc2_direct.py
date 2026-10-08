#!/usr/bin/env python
"""MeanVC2 backend直叩きによるRTF基準測定（numpy/torch/psutilのみ使用）。

scipy soundfile なしでも測れるよう、合成信号だけを使う。
品質には手を入れず、測定だけ。
結果は test-results/rtf_meanvc2_direct.json に保存する。
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vc.meanvc2 import MeanVC2Backend
from src.vc.models import VOICE_PROFILES, profile, is_meanvc2

SUPPORTED_MEANVC2 = [k for k in VOICE_PROFILES if is_meanvc2(k)]


def statistics(values, key):
    return float(np.mean([v[key] for v in values]))


def synthesize_audio(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(round(seconds * sr), dtype=np.float32) / sr
    voice = 0.15 * np.sin(2 * np.pi * 210 * t)
    vib = 1.0 + 0.6 * np.sin(2 * np.pi * 4.2 * t)
    voice *= vib
    ap = 0.6 + 0.4 * np.sin(2 * np.pi * 2.6 * t)
    rng = np.random.RandomState(7)
    noise = rng.randn(len(t)).astype(np.float32) * (ap * 0.02)
    return (voice + noise).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seconds', type=float, default=6.0)
    p.add_argument('--model', choices=SUPPORTED_MEANVC2, default='meanvc2_ref20')
    p.add_argument('--report', type=Path, default=ROOT / 'test-results/rtf_meanvc2_direct.json')
    args = p.parse_args()

    args.report.parent.mkdir(parents=True, exist_ok=True)

    report = {
        'stage': 'loading',
        'model': args.model,
        'threads': args.threads,
        'seconds': args.seconds,
        'input_kind': 'MeanVC2 backend direct process_chunk; no audio devices',
        'source': None,
        'source_is_synthetic': True,
    }

    def save(payload):
        args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
                                encoding='utf-8')

    save(report)

    backend = MeanVC2Backend(threads=args.threads)
    folder = ROOT / 'models' / profile(args.model)['folder']
    backend.load(folder)

    model_stats = backend.get_stats()
    report['source_name'] = 'synthetic_voice_210hz_with_slow_envelope'

    save({**report, 'stage': 'warmup', 'model': model_stats})
    backend.warmup()

    audio = synthesize_audio(args.seconds)
    n = math.ceil(len(audio) / 2560)
    audio = np.pad(audio, (0, n * 2560 - len(audio)))

    times = []
    stages = []
    saved = backend.get_stats()
    save({**report, 'stage': 'inference', 'model': saved})

    for i in range(n):
        begin = time.perf_counter()
        backend.process_chunk(audio[i * 2560:(i + 1) * 2560])
        times.append(time.perf_counter() - begin)
        stages.append(backend.stats['last_stage_ms'])

    delay = round((saved['algorithmic_buffer_ms'] + saved.get('interpolation_grid_delay_ms', 0)) * 16)
    tail_blocks = int(math.ceil(delay / 2560) + 8)
    direct_outputs = [backend.process_chunk(audio[i*2560:(i+1)*2560]) for i in range(n)]
    tail = [backend.process_chunk(np.zeros(2560, dtype=np.float32)) for _ in range(tail_blocks)]
    result = np.concatenate(direct_outputs + tail)
    result = result[delay:delay + len(audio)]

    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite direct MeanVC2 output')

    import psutil
    proc = psutil.Process()
    report.update({
        'stage': 'complete',
        'profile': args.model,
        'source': None,
        'source_is_synthetic': True,
        'source_name': report.get('source_name'),
        'audio_seconds': len(audio) / 16000,
        'padded_audio_seconds': len(audio) / 16000,
        'generation_seconds': float(sum(times)),
        'rtf': float(sum(times)) / (len(audio) / 16000),
        'chunk_mean_ms': float(np.mean(times) * 1000),
        'chunk_p95_ms': float(np.percentile(times, 95) * 1000),
        'chunk_max_ms': float(max(times) * 1000),
        'process_ram_bytes': proc.memory_info().rss,
        'output_finite': bool(np.isfinite(result).all()),
        'output_rms': float(np.sqrt(np.mean(result.astype('float64') ** 2))),
        'stage_mean_ms': {k: statistics(stages, k) for k in ('features', 'vc', 'vocoder')},
        'model_stats': model_stats,
        'input_kind': 'MeanVC2 backend direct process_chunk; no audio devices',
    })

    save(report)
    print(json.dumps({k: report[k] for k in ('stage', 'model', 'threads', 'rtf', 'chunk_mean_ms',
                                              'chunk_p95_ms', 'chunk_max_ms',
                                              'stage_mean_ms', 'process_ram_bytes',
                                              'source_is_synthetic', 'source_name')}, indent=2))
    backend.unload()


if __name__ == '__main__':
    main()
