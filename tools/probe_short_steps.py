"""Does raising the diffusion step count restore clarity on short utterances?

Renders the same real utterances at 2, 3 and 4 steps through the utterance route
with and without LavaSR, and reports the 4-8 kHz share and spectral tilt that
track intelligibility on a 16 kHz-bandlimited signal. Offline WAV only.
"""
import argparse, json, sys, wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vc.voice_library import DEFAULT_VOICE_ID

from src.vc.utterance import UtteranceCollector, convert_utterance, utterance_limit


def read48(path):
    with wave.open(str(path), 'rb') as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()),
                             dtype='<i2').astype(np.float32) / 32768


def collect(audio):
    collector = UtteranceCollector()
    found = []
    for start in range(0, len(audio), 7680):
        found.extend(collector.feed(audio[start:start + 7680]))
    tail = collector.flush()
    if tail is not None:
        found.append(tail)
    return found


def measure(x):
    x = x.astype('float64')
    x = x - x.mean()
    n = len(x)
    power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / 48000)

    def band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(power[mask].sum()) if mask.any() else 0.0

    low, high, very = band(100, 1000), band(4000, 8000), band(8000, 20000)
    total = low + band(1000, 4000) + high + very
    return {'tilt_4_8k_db': round(10 * np.log10(max(high, 1e-12) / max(low, 1e-12)), 2),
            'share_4_8k': round(high / total, 5),
            'share_8k_plus': round(very / total, 5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=ROOT / 'recordings/latency/ai_voice/Main_Input.wav')
    parser.add_argument('--model', default=DEFAULT_VOICE_ID)
    parser.add_argument('--report', type=Path,
                        default=ROOT / 'validation/v011/short_utterance_steps.json')
    args = parser.parse_args()

    from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
    from src.vc.models import profile
    from src.vc.resampler import StreamingResampler
    from src.vc.lavasr import LavaSR

    audio = read48(args.source)
    utterances = sorted(collect(audio), key=len)[:3]
    selected = profile(args.model)
    down = StreamingResampler(48000, 16000)
    up = StreamingResampler(16000, 48000)
    lava = LavaSR()

    rows = []
    for index, utterance in enumerate(utterances):
        label = f'utt{index}_{len(utterance) / 48000:.2f}s'
        for steps in (2, 3, 4):
            for enhancer in ('none', 'lavasr'):
                backend = MeanVC2PhraseBackend(threads=2)
                backend.load(ROOT / 'models' / selected['folder'])
                backend.select_profile(dict(selected, steps=steps))
                backend.warmup()
                try:
                    result = convert_utterance(backend, down, up, utterance,
                                                max_seconds=utterance_limit('lavasr'))
                finally:
                    backend.unload()
                if enhancer == 'lavasr':
                    result = lava.process(result)
                row = dict(case=label, steps=steps, enhancer=enhancer,
                           seconds=len(utterance) / 48000,
                           exact_length=len(result) == len(utterance),
                           finite=bool(np.isfinite(result).all()), **measure(result))
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    print('\n== share_4_8k by steps (none) ==')
    for label in sorted({r['case'] for r in rows}):
        line = [f'{r["steps"]}step' for r in rows
                if r['case'] == label and r['enhancer'] == 'none']
        print(f'  {label}: ' + '  '.join(line))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(source=str(args.source), model=args.model, rows=rows),
                                     indent=2, ensure_ascii=False), encoding='utf-8')
    print('saved', args.report)


if __name__ == '__main__':
    main()