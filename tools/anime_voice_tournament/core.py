"""Reproducible recipes, blind mapping and audio validation; no model imports."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'recordings/v010_tournament'
PITCHES = (1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.)
# Installed VST exposes 0.5 st native Formant steps, not 0.25 st.
FORMANTS = (-.5, 0., .5, 1., 1.5)
MODEL_FOLDERS = {'jvs': 'models/girl_01', 'character': 'models/character_01'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


@dataclass(frozen=True)
class Candidate:
    model: str
    voice: int
    pitch: float = 4.
    formant: float = 0.
    merge: tuple = ()  # Native speaker weights; never waveform mixing.

    def __post_init__(self):
        if self.model not in MODEL_FOLDERS:
            raise ValueError('Unknown model')
        if type(self.voice) is not int or not 0 <= self.voice < 100:
            raise ValueError('Invalid voice')
        for name, value, bound in [('pitch', self.pitch, 12), ('formant', self.formant, 2)]:
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > bound:
                raise ValueError('Invalid ' + name)
        if abs(self.pitch * 8 - round(self.pitch * 8)) > 1e-7:
            raise ValueError('Pitch must use native 0.125 st resolution')
        if abs(self.formant * 2 - round(self.formant * 2)) > 1e-7:
            raise ValueError('Formant must use installed native 0.5 st resolution')
        if self.merge:
            if not 2 <= len(self.merge) <= 3:
                raise ValueError('Merge needs 2 or 3 speakers')
            ids = []
            for voice, weight in self.merge:
                if type(voice) is not int or not 0 <= voice < 100 or not math.isfinite(weight) or weight <= 0:
                    raise ValueError('Invalid merge')
                ids.append(voice)
            if len(set(ids)) != len(ids) or abs(sum(w for _, w in self.merge) - 1) > 1e-6:
                raise ValueError('Merge weights must sum to one for distinct speakers')

    def recipe(self):
        value = asdict(self)
        value['pitch'] = float(self.pitch)
        value['formant'] = float(self.formant)
        value['merge'] = tuple((v,float(w)) for v,w in self.merge)
        return value

    def key(self):
        return hashlib.sha256(json.dumps(self.recipe(), sort_keys=True).encode()).hexdigest()[:16]


def read_audio(path):
    with wave.open(str(path), 'rb') as w:
        if w.getsampwidth() != 2 or w.getcomptype() != 'NONE':
            raise ValueError('PCM16 WAV required')
        if not 0 < w.getnframes() <= w.getframerate() * 60 or w.getnchannels() not in (1, 2):
            raise ValueError('Invalid or too long WAV')
        rate, channels = w.getframerate(), w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32).reshape(-1, channels) / 32768
    return audio.mean(axis=1), rate, channels


def inspect_audio(path, source=False):
    audio, rate, channels = read_audio(path)
    duration = len(audio) / rate
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if not np.isfinite(audio).all() or rms < 1e-5:
        raise ValueError('Nonfinite or silent WAV')
    if source and (rate != 48000 or channels != 1 or not 8 <= duration <= 15.01):
        raise ValueError('Source must be mono 48kHz and 8–15 seconds')
    return dict(sample_rate=rate, channels=channels, frames=len(audio), duration=duration,
                rms=rms, peak=float(np.max(abs(audio))),
                clip_count=int(np.sum(abs(audio) >= 32767 / 32768)), sha256=digest(path))


def normalize(audio):
    """One scalar per utterance, capped ±6 dB, no EQ/gate/dynamic compressor."""
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Invalid output')
    peak = float(np.max(abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-5:
        raise ValueError('Silent output')
    gain = max(.5, min(2., .08 / rms))
    gain = min(gain, .95 / max(peak, 1e-8))
    return (audio * gain).astype(np.float32), dict(gain=gain, raw_peak=peak, raw_rms=rms,
                                                raw_over_one=int(np.sum(abs(audio) >= 1)))


def write_audio(path, audio, rate=48000):
    if not np.isfinite(audio).all() or np.max(abs(audio)) > .999:
        raise ValueError('Unsafe output; must normalize before write')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.wav')
    with wave.open(str(temp), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.round(audio * 32767).astype('<i2').tobytes())
    temp.replace(path)


def blind_mapping(keys, seed=1010):
    keys = sorted(set(keys)); random.Random(seed).shuffle(keys)
    return {key: f'A{i+1:03}' for i, key in enumerate(keys)}


def cache_valid(entry, key, output):
    try:
        return entry.get('status') == 'ok' and entry.get('cache_key') == key and inspect_audio(output)['sha256'] == entry.get('output_sha256')
    except (OSError, ValueError, wave.Error, EOFError):
        return False


def load_inventory():
    import tomllib
    found = []
    for model, folder in MODEL_FOLDERS.items():
        base = ROOT / folder
        if not (base / 'runtime.json').is_file():
            continue
        runtime = json.loads((base / 'runtime.json').read_text(encoding='utf-8'))
        config = (base / runtime['model']).resolve()
        if not config.is_relative_to(base.resolve()) or not config.is_file():
            continue
        metadata = tomllib.loads(config.read_text(encoding='utf-8'))
        for voice, data in metadata['voice'].items():
            found.append(dict(model=model, voice=int(voice), name=data['name'],
                              average_pitch=data.get('average_pitch'), config=str(config)))
    return found
