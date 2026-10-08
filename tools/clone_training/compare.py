"""Offline native-model audition; no audio devices or realtime app changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def convert(plugin_path, model, source, rate, voice, pitch):
    import numpy as np
    from pedalboard import load_plugin
    from src.vc.vst_state import set_model_preset
    plugin = load_plugin(str(plugin_path))
    plugin.preset_data = set_model_preset(plugin.preset_data, model, voice, pitch)
    plugin.formant_shift_st = 0
    chunk = 1248
    for _ in range(12):
        plugin.process(np.zeros(chunk, dtype=np.float32), rate, buffer_size=chunk, reset=False)
    plugin.reset()
    output, timings = [], []
    # Include tail for the plugin's native latency, then remove that latency.
    latency = int(plugin.reported_latency_samples)
    padded = np.pad(source, (0, latency + chunk * 2))
    for start in range(0, len(padded), chunk):
        frame = np.zeros(chunk, dtype=np.float32)
        count = min(chunk, len(padded) - start)
        frame[:count] = padded[start:start + count]
        before = time.perf_counter()
        y = plugin.process(frame, rate, buffer_size=chunk, reset=False).reshape(-1)
        timings.append((time.perf_counter() - before) * 1000)
        if len(y) != chunk or not np.isfinite(y).all():
            raise ValueError('Invalid native model output')
        output.append(y.copy())
    result = np.concatenate(output)[latency:latency + len(source)]
    peak = float(np.max(np.abs(result)))
    # One scalar only; never clip away conversion defects or apply extra DSP.
    scale = min(1., .98 / max(peak, 1e-8))
    return result * scale, {'native_latency_samples':latency, 'peak_before_scale':peak,
                            'output_scale':scale, 'processing_p95_ms':float(np.percentile(timings, 95))}


def main():
    from tools.compare_presets import read_wav, write_wav
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'recordings/v0102_clone')
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    runtime = json.loads((ROOT / 'models/character_01/runtime.json').read_text(encoding='utf8'))
    plugin = ROOT / 'models/character_01' / runtime['plugin']
    baseline_runtime = json.loads((ROOT / 'models/girl_01/runtime.json').read_text(encoding='utf8'))
    baseline = ROOT / 'models/girl_01' / baseline_runtime['model']
    candidates = [('baseline_jvs002', baseline, 1, 4.)]
    for step in (() if args.baseline_only else (1000, 2500, 5000)):
        models = list((args.models / f'clone_{step}').rglob('*.toml'))
        if len(models) != 1:
            raise ValueError(f'Expected one exported model TOML at {step}, found {len(models)}')
        candidates.append((f'custom_{step}', models[0], 0, 0.))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source_path in enumerate(args.source, 1):
        source, rate = read_wav(source_path)
        if rate != 48000 or not 5 <= len(source) / rate <= 15:
            raise ValueError('Held-out input must be 48 kHz and 5–15 seconds')
        folder = args.output / f'input_{index:02d}'
        folder.mkdir(exist_ok=True)
        for name, model, voice, pitch in candidates:
            y, metrics = convert(plugin, model, source, rate, voice, pitch)
            path = folder / f'{name}.wav'
            write_wav(path, y, rate)
            records.append({'source':str(source_path), 'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
                            'candidate':name, 'model':str(model), 'pitch':pitch, 'voice':voice,
                            'output':str(path), 'prosody':False, 'extra_eq':False, **metrics})
    import shutil
    shutil.copy2(args.target, args.output / 'target_reference.wav')
    # Stable blind mapping lives separately from anonymous WAVs.
    blind = args.output / 'blind'
    blind.mkdir(exist_ok=True)
    mapping = []
    for index, record in enumerate(sorted(records, key=lambda r:hashlib.sha256((r['source_sha256'] + r['candidate']).encode()).hexdigest()), 1):
        name = f'A{index:02d}.wav'
        shutil.copy2(record['output'], blind / name)
        mapping.append({'blind_id':name, **record})
    (args.output / 'comparison_metadata.json').write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding='utf8')
    # Preserve human ratings on repeated conversions; never fabricate scores.
    ratings = args.output / 'ratings.csv'
    if not args.baseline_only and not ratings.exists():
        import csv
        with ratings.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['blind_id', 'target_similarity', 'female', 'natural',
                             'vowels', 'consonants', 'mechanical', 'male_residue', 'comment'])
            writer.writerows([[r['blind_id']] + [''] * 8 for r in mapping])
    print(f'Generated {len(records)} comparisons. Listening verdict remains pending.')


if __name__ == '__main__':
    main()
