"""Measure MeanVC2 runtime variants on a copied runtime.json (no model changes).

Compares each variant against the baseline output numerically (correlation/RMS)
on the same synthetic input, and records per-stage timing. Offline only.
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vc.voice_library import DEFAULT_VOICE_ID

from src.vc.meanvc2 import MeanVC2Backend
from src.vc.models import profile


def synthesize_audio(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(round(seconds * sr), dtype=np.float32) / sr
    voice = 0.15 * np.sin(2 * np.pi * 210 * t)
    vib = 1.0 + 0.6 * np.sin(2 * np.pi * 4.2 * t)
    voice *= vib
    ap = 0.6 + 0.4 * np.sin(2 * np.pi * 2.6 * t)
    rng = np.random.RandomState(7)
    noise = rng.randn(len(t)).astype(np.float32) * (ap * 0.02)
    return (voice + noise).astype(np.float32)


def run(folder: Path, threads: int, audio: np.ndarray, tail_blocks: int):
    backend = MeanVC2Backend(threads=threads)
    backend.load(folder)
    backend.warmup()
    times = []
    stages = []
    outputs = []
    for i in range(len(audio) // 2560):
        begin = time.perf_counter()
        outputs.append(backend.process_chunk(audio[i * 2560:(i + 1) * 2560]))
        times.append(time.perf_counter() - begin)
        stages.append(backend.stats['last_stage_ms'])
    for _ in range(tail_blocks):
        outputs.append(backend.process_chunk(np.zeros(2560, dtype=np.float32)))
    stats = backend.get_stats()
    backend.unload()
    return (float(np.mean(times) * 1000), float(np.percentile(times, 95) * 1000),
            float(max(times) * 1000),
            {k: float(np.mean([s[k] for s in stages])) for k in ('features', 'vc', 'vocoder')},
            np.concatenate(outputs), stats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seconds', type=float, default=6.0)
    p.add_argument('--report', type=Path, default=ROOT / 'test-results/rtf_variants.json')
    args = p.parse_args()

    source = ROOT / 'models' / profile(DEFAULT_VOICE_ID)['folder']
    base_runtime = json.loads((source / 'runtime.json').read_text('utf-8'))
    variants = {
        'baseline': {},
        'group6': {'vc_group_chunks': 6},
        'vocab36': {'vocoder_batch_frames': 36},
        'group6_vocab36': {'vc_group_chunks': 6, 'vocoder_batch_frames': 36},
        'group6_vocab36_aligned': {'vc_group_chunks': 6, 'vocoder_batch_frames': 36,
                                   'feature_frontend': 'aligned', 'bn_interpolation': 'fixed_linear'},
        'group6_aligned': {'vc_group_chunks': 6,
                           'feature_frontend': 'aligned', 'bn_interpolation': 'fixed_linear'},
    }

    audio = synthesize_audio(args.seconds)
    n = math.ceil(len(audio) / 2560)
    audio = np.pad(audio, (0, n * 2560 - len(audio)))
    # 480ms legacy buffer + 8 tail blocks, matching the direct tool's margin.
    delay_samples = round(480 * 16)
    tail_blocks = int(math.ceil(delay_samples / 2560)) + 8

    work = args.report.parent / 'rtf_variant_folders'
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    reference = None
    for name, overrides in variants.items():
        folder = work / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        runtime = dict(base_runtime)
        runtime.update(overrides)
        (folder / 'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False), 'utf-8')
        for asset in ('fixed_embedding.npy', 'reference.wav'):
            shutil.copy2(source / asset, folder / asset)
        mean_ms, p95_ms, max_ms, stage_ms, output, stats = run(folder, args.threads, audio, tail_blocks)
        trimmed = output[delay_samples:delay_samples + len(audio)]
        if reference is None:
            reference = trimmed
        corr = float(np.corrcoef(reference.astype(np.float64), trimmed.astype(np.float64))[0, 1])
        results[name] = dict(
            chunk_mean_ms=mean_ms, chunk_p95_ms=p95_ms, chunk_max_ms=max_ms,
            stage_mean_ms=stage_ms,
            rtf=None,  # set below from mean chunk time (ms per 160-sample block)
            output_rms=float(np.sqrt(np.mean(trimmed.astype(np.float64) ** 2))),
            correlation_vs_baseline=corr,
            config={k: stats.get(k) for k in ('feature_frontend', 'bn_interpolation',
                                              'vocoder_batch_frames', 'vc_group_chunks', 'steps')},
        )
        results[name]['rtf'] = float((mean_ms / 160.0))
        print(json.dumps({'variant': name, **{k: results[name][k] for k in
              ('chunk_mean_ms', 'chunk_p95_ms', 'rtf', 'output_rms', 'correlation_vs_baseline')}}, ensure_ascii=False))
    args.report.write_text(json.dumps(dict(threads=args.threads, variants=results), indent=2, ensure_ascii=False), 'utf-8')
    print('saved', args.report)


if __name__ == '__main__':
    main()
