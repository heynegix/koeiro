"""Diagnose why short utterances sound muffled in the utterance delivery mode.

Uses the app's own UtteranceDetector on a recorded physical-microphone sample, so
the utterances are the ones the app would really emit. Each utterance is rendered
through convert_utterance + LavaSR under four configurations that separate the
candidate causes: inference grouping (1 vs 6 blocks per call) and phrase repair
(pitch/energy TD-PSOLA on vs off). Offline WAV only: no audio device, no
listening, no training.
"""
import argparse, hashlib, json, sys, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vc.voice_library import DEFAULT_VOICE_ID

import numpy as np

from src.vc.utterance import UtteranceCollector

FRAME = 960  # 20 ms at 48 kHz, the same frame the collector uses


def read48(path):
    with wave.open(str(path), 'rb') as handle:
        if handle.getframerate() != 48000 or handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError('Expected 48 kHz mono 16-bit WAV')
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype='<i2').astype(np.float32) / 32768


def collect(audio):
    """Run the app's own detector and return the utterances it would emit."""
    collector = UtteranceCollector()
    found = []
    for start in range(0, len(audio), 7680):
        found.extend(collector.feed(audio[start:start + 7680]))
    tail = collector.flush()
    if tail is not None:
        found.append(tail)
    return found


def first_mora(audio, floor_db=-48.0, lead_ms=30, tail_ms=200):
    """Approximate one character: the first voiced run plus the app's natural tail."""
    threshold = 10 ** (floor_db / 20)
    frames = len(audio) // FRAME
    voiced = [float(np.sqrt(np.mean(audio[i * FRAME:(i + 1) * FRAME].astype(np.float64) ** 2))) >= threshold
              for i in range(frames)]
    start = None
    for index, value in enumerate(voiced):
        if value:
            start = index
            break
    if start is None:
        return None
    end = start
    while end + 1 < frames and voiced[end + 1]:
        end += 1
    first = max(0, start * FRAME - round(lead_ms / 1000 * 48000))
    last = min(len(audio), (end + 1) * FRAME + round(tail_ms / 1000 * 48000))
    return audio[first:last].copy()


def envelope(audio, floor_db=-50.0):
    frames = len(audio) // FRAME
    if frames == 0:
        return {}
    rms = np.array([float(np.sqrt(np.mean(audio[i * FRAME:(i + 1) * FRAME].astype(np.float64) ** 2)))
                    for i in range(frames)])
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 0:
        return {}
    threshold = peak * 10 ** (floor_db / 20)
    active = np.nonzero(rms >= threshold)[0]
    if active.size == 0:
        return {'voiced_span_ms': 0.0, 'leading_silence_ms': len(audio) / 48.0,
                'trailing_silence_ms': len(audio) / 48.0}
    span = (active[-1] - active[0] + 1) * 20.0
    return {'voiced_span_ms': span,
            'leading_silence_ms': float(active[0] * 20.0),
            'trailing_silence_ms': float((frames - 1 - active[-1]) * 20.0),
            'envelope_roughness_db': float(np.std(20 * np.log10(np.maximum(rms[active], 1e-9))))}


def spectrum(audio):
    """Tilt and high-band share: dull/muffled output shows a lower tilt."""
    if len(audio) < 2048:
        return {}
    x = audio.astype(np.float64)
    x = x - x.mean()
    window = np.hanning(len(x))
    spectrum_power = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / 48000)
    def band(low, high):
        mask = (freqs >= low) & (freqs < high)
        return float(spectrum_power[mask].sum()) if mask.any() else 0.0
    low = band(100, 1000)
    mid = band(1000, 4000)
    high = band(4000, 8000)
    very_high = band(8000, 20000)
    total = low + mid + high + very_high
    if total <= 0:
        return {}
    return {'tilt_1k_8k_db': round(10 * np.log10(max(high, 1e-12) / max(low, 1e-12)), 3),
            'band_share_0_1k': round(low / total, 4),
            'band_share_4_8k': round(high / total, 4),
            'band_share_8k_plus': round(very_high / total, 4)}


def render(backend_cls, selected, folder, audio, group, repair, down, up, enhancer):
    backend = backend_cls(threads=2)
    backend.load(ROOT / 'models' / folder)
    backend.select_profile(dict(selected, vc_group_chunks=group))
    backend.warmup()
    try:
        if repair is not None and getattr(backend, 'repair', None) is not None:
            backend.repair.energy = repair
            backend.repair.pitch = repair
        from src.vc.utterance import convert_utterance, utterance_limit
        result = convert_utterance(backend, down, up, audio,
                                   max_seconds=utterance_limit('lavasr'))
        stats = backend.get_stats()
        repair_stats = dict(stats.get('phrase', {})) if stats.get('phrase') else {}
        if enhancer is not None:
            result = enhancer.process(result)
        return result, stats, repair_stats
    finally:
        backend.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=ROOT / 'recordings/latency/ai_voice/Main_Input.wav')
    parser.add_argument('--model', default=DEFAULT_VOICE_ID)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--write-dir', type=Path,
                        default=ROOT / 'recordings/v011_short_utterance/blind')
    parser.add_argument('--report', type=Path,
                        default=ROOT / 'validation/v011/short_utterance_diagnosis.json')
    args = parser.parse_args()

    from src.vc.meanvc2 import MeanVC2Backend
    from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
    from src.vc.models import profile
    from src.vc.resampler import StreamingResampler
    from src.vc.lavasr import LavaSR

    audio = read48(args.source)
    utterances = collect(audio)
    if not utterances:
        raise SystemExit('no utterances detected in the sample')
    ordered = sorted(utterances, key=len)
    cases = []
    mora = first_mora(ordered[0])
    if mora is not None:
        cases.append(('one_mora', mora))
    for index, utterance in enumerate(ordered[:3]):
        cases.append((f'utt{index}_{len(utterance) / 48000:.2f}s', utterance))

    selected = profile(args.model)
    folder = selected['folder']
    enhancer = LavaSR()
    down = StreamingResampler(48000, 16000)
    up = StreamingResampler(16000, 48000)

    configurations = [
        ('group6_repair', MeanVC2PhraseBackend, 6, True),
        ('group1_repair', MeanVC2PhraseBackend, 1, True),
        ('group6_norepair', MeanVC2PhraseBackend, 6, False),
        ('group1_norepair', MeanVC2PhraseBackend, 1, False),
    ]

    args.write_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    rows = []
    for name, utterance in cases:
        for label, backend_cls, group, repair in configurations:
            result, stats, repair_stats = render(backend_cls, selected, folder, utterance,
                                                 group, repair, down, up, enhancer)
            row = dict(case=name, configuration=label, group=group, repair=repair,
                       input_seconds=len(utterance) / 48000,
                       output_seconds=len(result) / 48000,
                       exact_length=len(result) == len(utterance),
                       finite=bool(np.isfinite(result).all()),
                       peak=float(np.max(abs(result))) if result.size else 0.0,
                       rms=float(np.sqrt(np.mean(result.astype('float64') ** 2))) if result.size else 0.0,
                       algorithmic_buffer_ms=stats['algorithmic_buffer_ms'],
                       repair_alignment_ms=repair_stats.get('extra_delay_ms'),
                       repair_applied=repair_stats.get('applied'),
                       repair_late_bypass=repair_stats.get('late_bypass'),
                       repair_max_shift_st=repair_stats.get('max_shift_st'),
                       repair_max_gain_db=repair_stats.get('max_gain_db'),
                       **envelope(result), **spectrum(result))
            destination = args.write_dir / f'{name}__{label}.wav'
            sf.write(destination, result, 48000, subtype='FLOAT')
            row['output'] = str(destination)
            row['output_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
            rows.append(row)
            print(json.dumps({k: row[k] for k in
                              ('case', 'configuration', 'output_seconds', 'voiced_span_ms',
                               'leading_silence_ms', 'tilt_1k_8k_db', 'band_share_8k_plus',
                               'envelope_roughness_db', 'repair_applied', 'repair_max_shift_st')},
                             ensure_ascii=False), flush=True)

    summary = dict(
        input_kind='Recorded physical-microphone sample segmented by the app own UtteranceCollector',
        source=str(args.source), model=args.model, threads=args.threads,
        detection=dict(utterances=[len(u) / 48000 for u in utterances]),
        configurations=[c[0] for c in configurations], rows=rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('saved', args.report)


if __name__ == '__main__':
    main()