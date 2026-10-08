"""One isolated native render. No device, ASR, Prosody, gate or post FX."""
import argparse
import json
from pathlib import Path
import sys
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.anime_voice_tournament.core import Candidate, ROOT, MODEL_FOLDERS, read_audio, write_audio, normalize, save_json
from src.vc.beatrice_vst import BeatriceVSTBackend
from src.vc.vst_state import set_model_preset


def pair_cursor(weight):
    """Inverse-distance weights from official voice_morph_state.h, epsilon .0008."""
    lo, hi = .2, .8
    for _ in range(50):
        x = (lo + hi) / 2
        a, b = 1 / ((x-.2)**2+.0008)**2, 1 / ((x-.8)**2+.0008)**2
        if a / (a+b) > weight: lo = x
        else: hi = x
    x = round((lo+hi)/2, 3)  # installed cursor resolution
    a, b = 1 / ((x-.2)**2+.0008)**2, 1 / ((x-.8)**2+.0008)**2
    return x, a / (a+b)


def configure_merge(plugin, candidate, config):
    if len(candidate.merge) != 2:
        raise ValueError('Only verified native two-speaker Morph supported')
    count = len(tomllib.loads(config.read_text(encoding='utf-8'))['voice'])
    if any(v >= count for v, _ in candidate.merge):
        raise ValueError('Missing merge voice')
    plugin.morph_marker_count = 2
    plugin.morph_falloff = 2.
    for i, (voice, _) in enumerate(candidate.merge):
        setattr(plugin, f'morph_marker_{i}_voice', voice)
        setattr(plugin, f'morph_marker_{i}_x', .2 if i == 0 else .8)
        setattr(plugin, f'morph_marker_{i}_y', .5)
    x, actual = pair_cursor(candidate.merge[0][1])
    plugin.morph_cursor_x = x; plugin.morph_cursor_y = .5
    # Special final speaker slot selects native embedding Morph, not output sum.
    plugin.voice = 'ID ' + str(count)
    return dict(cursor=x, weights=[actual, 1-actual], native_morph=True,
                bitwise_deterministic=False, note='Native codebook Morph includes randomized selection; retain cache')


def render(job):
    spec = job['candidate'].copy(); spec['merge'] = tuple(tuple(x) for x in spec.get('merge', ()))
    candidate = Candidate(**spec)
    audio, rate, channels = read_audio(job['source'])
    if rate != 48000 or channels != 1:
        raise ValueError('Source must be mono 48 kHz')
    audio = audio[:int(job['seconds']*rate)]
    backend = BeatriceVSTBackend(chunk_factor=2)
    begin = time.perf_counter()
    try:
        backend.load(ROOT / MODEL_FOLDERS[candidate.model])
        voices = tomllib.loads(backend._config.read_text(encoding='utf-8'))['voice']
        if str(candidate.voice) not in voices:
            raise ValueError('Missing voice in installed model')
        p = backend.plugin
        p.preset_data = set_model_preset(p.preset_data, backend._config, candidate.voice, candidate.pitch)
        merge = configure_merge(p, candidate, backend._config) if candidate.merge else None
        p.formant_shift_st = candidate.formant
        # Fix pitch after model/voice/formant changes (native controller may relink them).
        p.pitch_shift_st = candidate.pitch
        p.input_gain_db = 0.; p.output_gain_db = 0.
        p.intonation_intensity = 1.; p.pitch_correction = 0.
        pitch, formant = float(p.pitch_shift_st), float(p.formant_shift_st)
        if abs(pitch-candidate.pitch) > 1e-6 or abs(formant-candidate.formant) > 1e-6:
            raise ValueError('Requested native parameter not applied')
        backend.warmup(); backend.reset()
        chunk = backend.chunk_samples
        delay = int(p.reported_latency_samples)
        if not 0 <= delay <= rate:
            raise ValueError('Invalid native reported latency')
        padded = np.pad(audio, (0, delay + chunk))
        pieces = []
        for pos in range(0, len(padded), chunk):
            frame = np.zeros(chunk, np.float32)
            length = min(chunk, len(padded)-pos); frame[:length] = padded[pos:pos+length]
            # Preserve raw native peaks for validation; do not use backend's clipped wrapper.
            y = p.process(frame, rate, buffer_size=chunk, reset=False).reshape(-1)
            if len(y) != chunk or not np.isfinite(y).all():
                raise ValueError('Invalid native output')
            pieces.append(y.copy())
        result = np.concatenate(pieces)[delay:delay+len(audio)]
        output, norm = normalize(result)
        write_audio(job['output'], output)
        stats = dict(generation_seconds=time.perf_counter()-begin, applied_pitch=pitch,
                     applied_formant=formant, reported_latency_samples=delay,
                     trimming='reported VST latency only; algorithmic residual not measured',
                     normalization=norm, merge=merge, prosody=False, post_fx=False,
                     output_frames=len(output), finite=True)
        save_json(job['result'], stats)
    finally:
        backend.unload()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job'); args = parser.parse_args()
    render(json.loads(Path(args.job).read_text(encoding='utf-8')))
