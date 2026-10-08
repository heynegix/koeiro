"""RTF harness for the utterance-delivery path with a post-VC bandwidth enhancer.

Offline WAV replay only: no audio devices, no listening, no training. Reports the
VC stage and the enhancer stage separately so each can be optimised from
measurement, and checks the exact-length / finite-output contract that the
utterance bridge depends on. The VC backend honours the selected profile so this
measures the real production route.
"""
import argparse, hashlib, json, math, sys, time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vc.voice_library import DEFAULT_VOICE_ID

import numpy as np
import psutil

SOURCES = {
    'normal': ROOT / 'recordings/v011_mega_tournament/source/source_normal.wav',
    'long': ROOT / 'recordings/v011_meanvc2_continuity/source/source_long.wav',
}


def read48(path, seconds):
    with wave.open(str(path), 'rb') as handle:
        assert handle.getframerate() == 48000 and handle.getsampwidth() == 2
        raw = handle.readframes(seconds * 48000)
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768


def build_enhancer(kind, threads):
    if kind == 'none':
        return None
    if kind == 'flashsr':
        from src.vc.flashsr import FlashSR
        return FlashSR(intra_op_threads=threads)
    if kind == 'lavasr':
        from src.vc.lavasr import LavaSR
        return LavaSR()
    raise ValueError('Unsupported enhancer: ' + kind)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=DEFAULT_VOICE_ID,
                   help='the shipped standard voice by default')
    p.add_argument('--vc-threads', type=int, default=4)
    p.add_argument('--enhancer', default='flashsr', choices=('flashsr', 'lavasr', 'none'))
    p.add_argument('--enhancer-threads', type=int, default=1)
    p.add_argument('--vc-group-chunks', type=int, choices=(1, 3, 6))
    p.add_argument('--cases', default='normal,long')
    p.add_argument('--seconds', default='8,25')
    p.add_argument('--limit-seconds', type=int,
                   help='Override the utterance limit so a long-form contract can be exercised.')
    p.add_argument('--tile-source', action='store_true',
                   help='Repeat the source to reach the requested length; length/memory contract only.')
    p.add_argument('--write-dir', type=Path,
                   help='Write each case result as a 48 kHz WAV for blind listening.')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()

    from src.vc.meanvc2 import MeanVC2Backend
    from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
    from src.vc.models import profile
    from src.vc.utterance import MAX_SECONDS, QUALITY_MAX_SECONDS, convert_utterance
    from src.vc.resampler import StreamingResampler

    selected = profile(args.model)
    backend_cls = MeanVC2PhraseBackend if selected.get('phrase_repair') else MeanVC2Backend
    backend = backend_cls(threads=args.vc_threads)
    backend.load(ROOT / 'models' / selected['folder'])
    backend.select_profile(selected)
    if args.vc_group_chunks is not None:
        override = dict(selected, vc_group_chunks=args.vc_group_chunks)
        backend.select_profile(override)
    backend.warmup()
    down = StreamingResampler(48000, 16000) if backend.sample_rate == 16000 else None
    up = StreamingResampler(16000, 48000) if backend.sample_rate == 16000 else None
    enhancer = build_enhancer(args.enhancer, args.enhancer_threads)
    stats = backend.get_stats()
    limit = QUALITY_MAX_SECONDS if args.enhancer in ('mossformer', 'voicefixer') else MAX_SECONDS
    if args.limit_seconds is not None:
        if not 1 <= args.limit_seconds <= QUALITY_MAX_SECONDS:
            raise SystemExit(f'--limit-seconds must be 1..{QUALITY_MAX_SECONDS}')
        limit = args.limit_seconds

    process = psutil.Process()
    peak = process.memory_info().rss
    rows = []
    cases = args.cases.split(',')
    seconds = [int(x) for x in args.seconds.split(',')]
    for name, want in zip(cases, seconds):
        audio = read48(SOURCES[name], want)
        if args.tile_source and want * 48000 > len(audio):
            repeats = -(-(want * 48000) // len(audio))
            audio = np.tile(audio, repeats)
        audio = audio[:min(len(audio), limit * 48000)]
        for attempt in range(args.repeat):
            t0 = time.monotonic()
            result = convert_utterance(backend, down, up, audio, max_seconds=limit)
            vc_seconds = time.monotonic() - t0
            peak = max(peak, process.memory_info().rss)
            enhancer_seconds = 0.0
            enhancer_error = ''
            if enhancer is not None:
                t1 = time.monotonic()
                try:
                    result = enhancer.process(result)
                except Exception as error:
                    enhancer_error = repr(error)
                enhancer_seconds = time.monotonic() - t1
                peak = max(peak, process.memory_info().rss)
            total = time.monotonic() - t0
            audio_seconds = len(audio) / 48000
            row = dict(
                case=name, attempt=attempt, audio_seconds=audio_seconds,
                vc_seconds=round(vc_seconds, 3), enhancer_seconds=round(enhancer_seconds, 3),
                total_seconds=round(total, 3),
                vc_rtf=round(vc_seconds / audio_seconds, 4),
                enhancer_rtf=round(enhancer_seconds / audio_seconds, 4),
                rtf=round(total / audio_seconds, 4),
                exact_length=len(result) == len(audio),
                finite=bool(np.isfinite(result).all()),
                peak=float(np.max(np.abs(result))) if result.size else 0.0,
                rms=float(np.sqrt(np.mean(result.astype('float64') ** 2))) if result.size else 0.0,
                peak_ram_bytes=peak, enhancer_error=enhancer_error)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if args.write_dir:
                import soundfile as sf
                args.write_dir.mkdir(parents=True, exist_ok=True)
                destination = args.write_dir / f'{name}_{attempt}.wav'
                sf.write(destination, result, 48000, subtype='FLOAT')
                row['output'] = str(destination)

    summary = dict(
        input_kind='Offline WAV replay through convert_utterance; no audio devices',
        model=args.model, profile_folder=selected['folder'],
        vc_threads=args.vc_threads, enhancer=args.enhancer,
        enhancer_threads=args.enhancer_threads if args.enhancer != 'none' else None,
        vc_group_chunks=stats['vc_group_chunks'],
        vocoder_batch_frames=stats['vocoder_batch_frames'],
        feature_frontend=stats['feature_frontend'],
        bn_interpolation=stats['bn_interpolation'],
        algorithmic_buffer_ms=stats['algorithmic_buffer_ms'],
        phrase_repair=bool(stats.get('phrase_repair')),
        utterance_limit_seconds=limit, rows=rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('saved', args.report)
    backend.unload()


if __name__ == '__main__':
    main()