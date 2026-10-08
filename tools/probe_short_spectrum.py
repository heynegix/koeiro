"""Where does the high-frequency loss come from on short utterances?

Compares, per utterance, the recorded source, MeanVC2 alone, and MeanVC2+LavaSR,
using band shares and a spectral ceiling. Offline WAV only.
"""
import sys, wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vc.voice_library import DEFAULT_VOICE_ID

from src.vc.utterance import UtteranceCollector, convert_utterance, utterance_limit
from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
from src.vc.models import profile
from src.vc.resampler import StreamingResampler
from src.vc.lavasr import LavaSR


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


def bands(x):
    x = x.astype('float64')
    x = x - x.mean()
    n = len(x)
    power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / 48000)

    def band(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(power[mask].sum()) if mask.any() else 0.0

    low, mid, high, very = band(100, 1000), band(1000, 4000), band(4000, 8000), band(8000, 20000)
    total = low + mid + high + very
    reference = band(1000, 1100)
    ceiling = 0
    for k in range(8, 25):
        if band(k * 1000, (k + 1) * 1000) > reference * 1e-4:
            ceiling = k + 1
    return {'tilt': round(10 * np.log10(max(high, 1e-12) / max(low, 1e-12)), 2),
            's4_8k': round(high / total, 5),
            's8k': round(very / total, 5),
            'ceiling': ceiling,
            'rms': float(np.sqrt(np.mean(x ** 2)))}


def main():
    audio = read48(ROOT / 'recordings/latency/ai_voice/Main_Input.wav')
    utterances = sorted(collect(audio), key=len)[:3]
    selected = profile(DEFAULT_VOICE_ID)
    down = StreamingResampler(48000, 16000)
    up = StreamingResampler(16000, 48000)
    lava = LavaSR()
    print(f"{'case':<13}{'variant':<9}{'tilt_dB':>9}{'4_8k':>10}{'8k+':>10}{'ceil_kHz':>10}{'rms':>10}")
    for index, utterance in enumerate(utterances):
        label = f'utt{index}_{len(utterance) / 48000:.2f}s'
        variants = [('source', utterance)]
        for name in ('none', 'lavasr'):
            backend = MeanVC2PhraseBackend(threads=2)
            backend.load(ROOT / 'models' / selected['folder'])
            backend.select_profile(selected)
            backend.warmup()
            try:
                result = convert_utterance(backend, down, up, utterance,
                                            max_seconds=utterance_limit('lavasr'))
            finally:
                backend.unload()
            if name == 'lavasr':
                result = lava.process(result)
            variants.append((name, result))
        for name, signal in variants:
            m = bands(signal)
            print(f"{label:<13}{name:<9}{m['tilt']:>9}{m['s4_8k']:>10}{m['s8k']:>10}"
                  f"{m['ceiling']:>10}{m['rms']:>10.5f}")


if __name__ == '__main__':
    main()