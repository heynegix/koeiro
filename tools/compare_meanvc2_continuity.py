"""Resume bounded offline comparisons of the currently selected voice on D:."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'recordings/v011_meanvc2_continuity'
VARIANTS = {
    'current': (1, 1, 2),
    'decode360': (1, 36, 2),
    'vc360': (3, 1, 2),
    'vc720': (6, 1, 2),
    'vc720_steps3': (6, 1, 3),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def require_idle_voice_worker():
    """Inspect existing service load; never stop the user's application."""
    import psutil
    services = []
    for process in psutil.process_iter(['cmdline']):
        command = process.info.get('cmdline') or []
        if 'src.vc.service' in command:
            process.cpu_percent(None)
            services.append(process)
    if not services:
        return
    time.sleep(1)
    for process in services:
        try:
            if process.cpu_percent(None) > 20:
                raise RuntimeError('Live voice conversion is active. Press Stop in the app before offline comparison.')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def prepare_source(origin):
    from tools.vc_tournament.audio import stats
    folder = OUT / 'source'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / 'source_long.wav'
    stamp = destination.with_suffix('.origin.json')
    if origin:
        origin = origin.resolve(strict=True)
        original_hash = sha(origin)
        if not (stamp.exists() and destination.exists()
                and read(stamp).get('origin_sha256') == original_hash
                and read(stamp).get('wav_sha256') == sha(destination)):
            archived = folder / ('original_' + original_hash[:12] + origin.suffix)
            shutil.copy2(origin, archived)
            temporary = folder / 'source_long.partial.wav'
            ffmpeg = shutil.which('ffmpeg')
            if not ffmpeg:
                raise RuntimeError('Existing FFmpeg is required to decode the new recording')
            command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                       '-i', str(archived), '-map', '0:a:0', '-vn', '-ac', '1',
                       '-ar', '48000', '-c:a', 'pcm_s16le', str(temporary)]
            subprocess.run(command, check=True, timeout=60)
            info = stats(temporary)
            if not 15 <= info['duration'] <= 45:
                raise ValueError('Long source must be 15–45 seconds')
            temporary.replace(destination)
            save(stamp, dict(origin=str(origin), origin_sha256=original_hash,
                             archived_original=str(archived), wav_sha256=sha(destination),
                             command=command, stats=info,
                             processing='48kHz mono PCM16 decode only; no voice processing'))
    if not destination.exists():
        raise FileNotFoundError('Provide --source-long with the recorded long sentence')
    for source in (ROOT / 'recordings/v011_mega_tournament/source').glob('source_*.wav'):
        shutil.copy2(source, folder / source.name)


def worker(args):
    import numpy as np
    import psutil
    import soundfile as sf
    from scipy.signal import resample_poly
    from src.vc.meanvc2_continuity import MeanVC2ContinuityBackend
    p = psutil.Process()
    if os.name == 'nt':
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        p.cpu_affinity(p.cpu_affinity()[:args.threads])
    chunks, frames, steps = VARIANTS[args.variant]
    backend = MeanVC2ContinuityBackend(args.threads, chunks, frames)
    backend.load(ROOT / 'models' / args.model)
    # Changing the solver count is experimental; original embedding never changes.
    backend.steps = steps
    backend.stats['steps'] = steps
    backend.timesteps = [(backend.torch.tensor([1 - i / steps]),
                          backend.torch.tensor([1 - (i + 1) / steps])) for i in range(steps)]
    failed = False
    for source in sorted((OUT / 'source').glob('source_*.wav')):
        dest = OUT / 'raw' / args.variant / source.name
        stamp = dest.with_suffix('.json')
        config = dict(schema=1, model=args.model, variant=args.variant,
                      threads=args.threads, vc_chunks=chunks, decode_frames=frames, steps=steps,
                      source_sha256=sha(source), runtime_sha256=sha(ROOT / 'models' / args.model / 'runtime.json'),
                      embedding_sha256=backend.stats['fixed_embedding_sha256'],
                      backend_sha256=sha(ROOT / 'src/vc/meanvc2.py'),
                      experiment_sha256=sha(ROOT / 'src/vc/meanvc2_continuity.py'),
                      tool_sha256=sha(__file__))
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if stamp.exists() and dest.exists() and read(stamp).get('signature') == signature and read(stamp).get('output_sha256') == sha(dest):
            print('CACHE', args.variant, source.name, flush=True)
            continue
        if stamp.exists():
            shutil.copy2(stamp, OUT / 'metadata' / 'attempt_history' / (args.variant + '_' + source.stem + '_' + str(time.time_ns()) + '.json'))
        save(stamp, dict(status='RUNNING', config=config, signature=signature))
        try:
            backend.reset()
            backend.warmup()
            audio, sr = sf.read(source, dtype='float32')
            if sr != 48000 or audio.ndim != 1:
                raise ValueError('Expected prepared 48kHz mono')
            audio = resample_poly(audio, 1, 3).astype(np.float32)
            length = len(audio)
            audio = np.pad(audio, (0, (-length) % 2560))
            outputs, timings, stages = [], [], []
            start_cpu = time.process_time()
            start = time.perf_counter()
            for i in range(0, len(audio), 2560):
                began = time.perf_counter()
                outputs.append(backend.process_chunk(audio[i:i + 2560]))
                timings.append(time.perf_counter() - began)
                stages.append(dict(backend.stats['last_stage_ms']))
            delay = round((backend.algorithmic_buffer_ms + backend.stats['interpolation_grid_delay_ms']) * 16)
            # Extra zero input flushes grouped conversion and the vocoder's future context.
            for _ in range(math.ceil(delay / 2560) + chunks + 8):
                outputs.append(backend.process_chunk(np.zeros(2560, dtype=np.float32)))
            elapsed = time.perf_counter() - start
            result = np.concatenate(outputs)[delay:delay + length]
            if len(result) != length or not np.isfinite(result).all():
                raise RuntimeError('Invalid finite-stream alignment')
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, result, 16000, subtype='FLOAT')
            row = dict(status='SUCCESS', config=config, signature=signature,
                       source=str(source), output=str(dest), output_sha256=sha(dest),
                       audio_seconds=length / 16000, output_seconds=len(result) / 16000,
                       generation_seconds=sum(timings), rtf=sum(timings) / (length / 16000),
                       generation_with_flush_seconds=elapsed,
                       rtf_with_flush=elapsed / (length / 16000),
                       chunk_p95_ms=float(np.percentile(timings, 95) * 1000),
                       chunk_max_ms=max(timings) * 1000, model_buffer_ms=backend.algorithmic_buffer_ms,
                       interpolation_grid_delay_ms=backend.stats['interpolation_grid_delay_ms'],
                       process_ram_bytes=p.memory_info().rss, cpu_seconds=time.process_time() - start_cpu,
                       stage_mean_ms={k:float(np.mean([v[k] for v in stages])) for k in stages[0]},
                       clipping_samples=int(np.sum(np.abs(result) >= .999)),
                       input_kind='Offline WAV replay; no audio devices', human_quality_accepted=False)
            save(stamp, row)
            print('SUCCESS', args.variant, source.name, 'RTF', round(row['rtf'], 3), flush=True)
        except Exception as e:
            failed = True
            save(stamp, dict(status='FAILED', config=config, signature=signature, error=str(e)))
            print('FAILED', args.variant, source.name, str(e), flush=True)
    backend.unload()
    return 1 if failed else 0


def build():
    import numpy as np
    import soundfile as sf
    from tools.vc_tournament.audio import normalize
    rows = []
    for p in sorted((OUT / 'raw').glob('*/*.json')):
        row = read(p)
        if row.get('status') == 'SUCCESS' and Path(row['output']).is_file() and sha(row['output']) == row['output_sha256']:
            rows.append(row)
    # Decode grouping should preserve waveform, apart from finite FP32 rounding.
    parity = []
    for row in rows:
        if row['config']['variant'] != 'decode360':
            continue
        original = OUT / 'raw/current' / Path(row['source']).name
        if original.exists():
            a, sr = sf.read(original, dtype='float32')
            b, _ = sf.read(row['output'], dtype='float32')
            error = float(np.max(np.abs(a - b))) if len(a) == len(b) else None
            parity.append(dict(source=Path(row['source']).name, max_absolute_error=error,
                               passed=error is not None and error <= 1e-5))
    save(OUT / 'metadata/decode_equivalence.json', dict(rows=parity,
         passed=len(parity) == 4 and all(r['passed'] for r in parity)))
    shuffled = rows.copy()
    random.Random(114).shuffle(shuffled)
    public, private = [], []
    blind = OUT / 'blind'
    blind.mkdir(parents=True, exist_ok=True)
    baseline = OUT / 'baseline'
    baseline.mkdir(exist_ok=True)
    for source in (OUT / 'raw/current').glob('source_*.wav'):
        normalize(source, baseline / source.name)
    (OUT / 'reference').mkdir(exist_ok=True)
    for i, row in enumerate(shuffled, 1):
        ident = f'D{i:03}'
        dest = blind / (ident + '.wav')
        try:
            correction = normalize(row['output'], dest)
        except Exception as error:
            save(OUT / 'metadata' / ('blind_failure_' + row['config']['variant'] + '_' + Path(row['source']).stem + '.json'),
                 dict(status='FAILED', stage='blind_normalization', source=row['output'], error=str(error)))
            continue
        public.append(dict(id=ident, audio=dest.name, source='../source/' + Path(row['source']).name,
                           reference='../reference/current_reference.wav', sha256=sha(dest)))
        private.append(dict(id=ident, config=row['config'], normalization=correction))
    package = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
    save(OUT / 'metadata/blind_manifest.json', dict(package=package, candidates=private))
    template = (ROOT / 'tools/vc_tournament/listening.html').read_text('utf-8')
    template = template.replace('__DATA__', json.dumps(dict(package=package, candidates=public), ensure_ascii=False).replace('<', '\\u003c'))
    template = template.replace('v0.11 Blind Voice Comparison', '今の声を保つ・長文比較')
    template = template.replace("[['Candidate',c.audio],['Target Reference',c.reference],['Source',c.source]]",
                                "[['Candidate',c.audio],['現在のアプリの声','../baseline/'+c.source.split('/').pop()],['Target Reference',c.reference],['Source',c.source]]")
    template = template.replace("['mechanical','機械感']", "['mechanical','機械感'],['blur','声のぼやけ'],['voice_changed','今の声からの変化']")
    template = template.replace('元声残り・機械感：5が多い', '元声残り・機械感・ぼやけ・声の変化：5が多い')
    template = template.replace('v011-ratings.json', 'meanvc2-continuity-ratings.json')
    template = template.replace('<main id="candidates">', '<label>Source <select id="sourcefilter"><option value="long">長文</option><option value="normal">通常声</option><option value="low">低め</option><option value="bright">明るめ</option><option value="">全て</option></select></label><main id="candidates">')
    template = template.replace('for(const c of data.candidates){', "for(const c of data.candidates.filter(c=>!document.getElementById('sourcefilter').value||c.source.endsWith('source_'+document.getElementById('sourcefilter').value+'.wav'))){")
    template = template.replace('render();\n</script>', "document.getElementById('sourcefilter').onchange=render;\nrender();\n</script>")
    (blind / 'index.html').write_text(template, 'utf-8')
    save(OUT / 'metadata/results.json', dict(package=package, candidates=len(public), rows=rows))
    lines = ['# 今の声を保つ長文比較', '', 'Status: READY FOR HUMAN LISTENING' if len(public) == 20 else 'Status: PARTIAL RESULTS', '',
             '同一MeanVC2 checkpoint・Reference・固定Embedding。Prosody / Text-Aware / Pitch / Formant / EQなし。比較候補の生成中はアプリの選択・runtimeを変更していない。後から行った波形一致の効率化の採用・実機検証はmeanvc2_continuity_implementation_report.mdを参照。ぼやけ改善は人間評価待ち。', '',
             'Offline WAV replay。音声デバイスを開いていない。RTFは指定threads/affinityでの測定であり、実機End-to-End遅延は未測定。モデルbuffer＋grid delay＋既存1280ms pre-rollの合計は設計値のみ。', '',
             'decode360は同じmelをまとめて復元する効率化。vc360/vc720は同じ120ms attention maskを維持して複数区間を一緒に推論する実験で、声質が変わり得る。steps3も人間評価前に採用しない。', '',
             '| 条件 | Source | threads | RTF | モデルbuffer ms | chunk max ms |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['config']['variant']} | {Path(row['source']).stem} | {row['config']['threads']} | {row['rtf']:.3f} | {row['model_buffer_ms']} | {row['chunk_max_ms']:.1f} |")
    lines += ['', '## Decode waveform equivalence', '', json.dumps(parity, ensure_ascii=False, indent=2), '',
              'Failed / interrupted jobs are retained in raw stamps, metadata/*_process.json and metadata/attempt_history. Successful WAVs are SHA-verified on resume.', '']
    (ROOT / 'validation/v011/meanvc2_continuity_report.md').write_text('\n'.join(lines), 'utf-8')
    print(json.dumps(dict(candidates=len(public), page=str(blind / 'index.html'), decode_parity=parity)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-long', type=Path)
    p.add_argument('--model', choices=('meanvc2_ref20', 'meanvc2_ref60', 'meanvc2_120'))
    p.add_argument('--threads', type=int, choices=(1, 2, 4), default=2)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if args.worker:
        return worker(args)
    import psutil
    from tools.vc_tournament.process import run
    args.model = args.model or read(ROOT / 'settings.json')['ai_model']
    if args.model not in ('meanvc2_ref20', 'meanvc2_ref60', 'meanvc2_120'):
        raise ValueError('Current voice is not MeanVC2; specify the preserved MeanVC2 profile')
    for folder in ('metadata/attempt_history', 'logs', 'raw', 'reference'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    snapshot = dict(model=args.model, settings_sha256=sha(ROOT / 'settings.json'),
                    runtime_sha256=sha(ROOT / 'models' / args.model / 'runtime.json'),
                    reference_sha256=sha(ROOT / 'models' / args.model / 'reference.wav'),
                    embedding_sha256=sha(ROOT / 'models' / args.model / 'fixed_embedding.npy'))
    if not args.report_only:
        require_idle_voice_worker()
        save(OUT / 'metadata/preserved_voice.json', snapshot)
        prepare_source(args.source_long)
        shutil.copy2(ROOT / 'models' / args.model / 'reference.wav', OUT / 'reference/current_reference.wav')
        for variant in ([args.variant] if args.variant else VARIANTS):
            require_idle_voice_worker()
            if psutil.virtual_memory().available < 4 * 1024**3:
                save(OUT / 'metadata' / (variant + '_process.json'), dict(status='FAILED', reason='FREE_RAM_BELOW_4_GIB'))
                continue
            command = [ROOT / 'vc_models/meanvc2/.venv/Scripts/python.exe', Path(__file__),
                       '--worker', '--model', args.model, '--threads', str(args.threads), '--variant', variant]
            process = run(command, ROOT, OUT / 'logs' / (variant + '.log'), timeout=420, ram_limit_gb=4.5)
            target = OUT / 'metadata' / (variant + '_process.json')
            if target.exists():
                shutil.copy2(target, OUT / 'metadata/attempt_history' / (variant + '_process_' + str(time.time_ns()) + '.json'))
            save(target, process)
            print(variant, process['status'], round(process['wall_seconds'], 1), flush=True)
            build()
    build()
    if not args.report_only:
        preserved = read(OUT / 'metadata/preserved_voice.json')
        save(OUT / 'metadata/production_unchanged.json', dict(before=preserved, after=snapshot,
             files_still_identical=all(sha(path) == preserved[key] for key, path in (
                 ('settings_sha256', ROOT / 'settings.json'),
                 ('runtime_sha256', ROOT / 'models' / preserved['model'] / 'runtime.json'),
                 ('reference_sha256', ROOT / 'models' / preserved['model'] / 'reference.wav'),
                 ('embedding_sha256', ROOT / 'models' / preserved['model'] / 'fixed_embedding.npy')))))


if __name__ == '__main__':
    sys.exit(main() or 0)
