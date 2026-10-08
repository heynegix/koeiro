"""Render approved vs optimized MeanVC2 streaming conditions and compare them.

Offline WAV replay through the production backend only. No audio devices, no
training, no downloads. Each condition is resumed from its saved report so a
long sweep can be re-entered, and every report keeps the per-stage timing and
RAM evidence used to decide the optimisation.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import soundfile as sf
from scipy.signal import resample_poly

from src.vc.meanvc2 import MeanVC2Backend
from src.vc.models import VOICE_PROFILES, profile, is_meanvc2

SOURCES = {
    'normal': ROOT / 'recordings/v011_mega_tournament/source/source_normal.wav',
    'low': ROOT / 'recordings/v011_mega_tournament/source/source_low.wav',
    'bright': ROOT / 'recordings/v011_mega_tournament/source/source_bright.wav',
    'long': ROOT / 'recordings/v011_post_vc2/source/long.wav',
}
PAIRS = [
    ('meanvc2_ref20', 'meanvc2_ref20_optimized'),
    ('meanvc2_120', 'meanvc2_120_optimized'),
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load16k(path):
    audio, rate = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        audio = resample_poly(audio, 16000, rate).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def render(name, source, threads):
    audio = load16k(source)
    backend = MeanVC2Backend(threads=threads)
    backend.load(ROOT / 'models' / profile(name)['folder'])
    backend.select_profile(profile(name))
    backend.warmup()
    stats = backend.get_stats()
    delay = round((stats['algorithmic_buffer_ms']
                   + stats.get('interpolation_grid_delay_ms', 0)) * 16)
    n = (len(audio) + 2559) // 2560
    padded = np.pad(audio, (0, n * 2560 - len(audio)))
    chunks, times, stages = [], [], []
    process = psutil.Process()
    peak_ram = process.memory_info().rss
    with backend.torch.inference_mode():
        for i in range(n):
            begin = time.perf_counter()
            chunks.append(backend.process_chunk(padded[i * 2560:(i + 1) * 2560]))
            times.append(time.perf_counter() - begin)
            stages.append(backend.stats['last_stage_ms'])
            peak_ram = max(peak_ram, process.memory_info().rss)
        for _ in range(math.ceil(delay / 2560) + 8):
            chunks.append(backend.process_chunk(np.zeros(2560, dtype=np.float32)))
    stream = np.concatenate(chunks)
    result = stream[delay:delay + len(audio)]
    if len(result) != len(audio):
        raise RuntimeError('Incomplete finite-stream tail')
    report = {
        'stage': 'complete', 'input_kind': 'Offline WAV replay; no audio devices',
        'profile': name, 'source': str(source), 'threads': threads,
        'audio_seconds': len(audio) / 16000,
        'generation_seconds': float(np.sum(times)),
        'rtf': float(np.sum(times)) / (len(audio) / 16000),
        'chunk_mean_ms': float(np.mean(times) * 1000),
        'chunk_p95_ms': float(np.percentile(times, 95) * 1000),
        'chunk_max_ms': float(np.max(times) * 1000),
        'stage_mean_ms': {k: float(np.mean([s[k] for s in stages]))
                          for k in ('features', 'vc', 'vocoder')},
        'peak_ram_bytes': peak_ram,
        'algorithmic_buffer_ms': stats['algorithmic_buffer_ms'],
        'interpolation_grid_delay_ms': stats.get('interpolation_grid_delay_ms', 0),
        'vc_group_chunks': stats['vc_group_chunks'],
        'vocoder_batch_frames': stats['vocoder_batch_frames'],
        'steps': stats['steps'],
        'feature_frontend': stats['feature_frontend'],
        'bn_interpolation': stats['bn_interpolation'],
        'fixed_embedding_sha256': stats['fixed_embedding_sha256'],
        'reference_sha256': stats['reference_sha256'],
        'warmup_seconds': stats.get('warmup_seconds'),
        'output_rms': float(np.sqrt(np.mean(result.astype('float64') ** 2))),
        'output_finite': bool(np.isfinite(result).all()),
    }
    backend.unload()
    return report, result


def compare(reference, candidate):
    a = reference.astype(np.float64)
    b = candidate.astype(np.float64)
    diff = b - a
    correlation = float(np.corrcoef(a, b)[0, 1])
    rms = float(np.sqrt(np.mean(diff ** 2)))
    peak = float(np.max(np.abs(diff)))
    return {
        'max_abs_diff': peak,
        'rms_diff': rms,
        'correlation': correlation,
        'snr_db': float(10 * np.log10(np.mean(a ** 2) / max(np.mean(diff ** 2), 1e-30))),
        'correlation_above_0_999': correlation > 0.999,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--report-only', action='store_true',
                   help='Rebuild the comparison and blind page from cached WAVs.')
    args = p.parse_args()

    out = ROOT / 'recordings/v011_meanvc2_optimized'
    raw = out / 'raw'
    private = out / 'metadata'
    for folder in (raw, private):
        folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for source_name, source_path in SOURCES.items():
        for approved, optimized in PAIRS:
            for name in (approved, optimized):
                assert is_meanvc2(name), name
                report_path = raw / f'{name}_{source_name}.json'
                wav_path = raw / f'{name}_{source_name}.wav'
                if args.report_only:
                    if not (wav_path.exists() and report_path.exists()):
                        raise RuntimeError(f'missing cached {name}/{source_name}')
                    rows.append(json.loads(report_path.read_text('utf-8')))
                    continue
                if wav_path.exists() and report_path.exists():
                    rows.append(json.loads(report_path.read_text('utf-8')))
                    continue
                report, wave = render(name, source_path, args.threads)
                sf.write(wav_path, wave, 16000, subtype='FLOAT')
                report['input_sha256'] = sha(source_path)
                report['output_sha256'] = sha(wav_path)
                report['output'] = str(wav_path)
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                       encoding='utf-8')
                rows.append(report)
                print(json.dumps({'profile': name, 'source': source_name,
                                  'rtf': round(report['rtf'], 4),
                                  'p95': round(report['chunk_p95_ms'], 1),
                                  'buffer_ms': report['algorithmic_buffer_ms']},
                                 ensure_ascii=False), flush=True)

    summary = {'threads': args.threads, 'sources': list(SOURCES), 'rows': rows,
               'comparison': [], 'package': None}
    for source_name in SOURCES:
        for approved, optimized in PAIRS:
            a = raw / f'{approved}_{source_name}.wav'
            b = raw / f'{optimized}_{source_name}.wav'
            ja = json.loads((raw / f'{approved}_{source_name}.json').read_text('utf-8'))
            jb = json.loads((raw / f'{optimized}_{source_name}.json').read_text('utf-8'))
            diff = compare(sf.read(a, dtype='float32')[0], sf.read(b, dtype='float32')[0])
            diff.update(source=source_name, approved=approved, optimized=optimized,
                        approved_rtf=ja['rtf'], optimized_rtf=jb['rtf'],
                        rtf_ratio=ja['rtf'] / jb['rtf'],
                        approved_buffer_ms=ja['algorithmic_buffer_ms'],
                        optimized_buffer_ms=jb['algorithmic_buffer_ms'],
                        same_reference=ja['reference_sha256'] == jb['reference_sha256'],
                        same_embedding=ja['fixed_embedding_sha256'] == jb['fixed_embedding_sha256'])
            summary['comparison'].append(diff)

    (private / 'optimized_comparison.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    for row in summary['comparison']:
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in row.items()}, ensure_ascii=False))
    print('saved', private / 'optimized_comparison.json')


if __name__ == '__main__':
    main()