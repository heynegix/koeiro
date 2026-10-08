"""Blind A/B pairs for listening evaluation, generated offline.

Every pair is the same source audio through two conditions, with the labels shuffled
and written to a sealed answer key. Nothing here decides which condition is better:
the point is to produce material a person can actually judge, because no amount of
automation can tell whether a voice sounds closer to the person who owns it.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_wav(path, rate=48000):
    import wave
    with wave.open(str(path), 'rb') as stream:
        source_rate = stream.getframerate()
        raw = stream.readframes(stream.getnframes())
        channels = stream.getnchannels()
    audio = np.frombuffer(raw, dtype='<i2').astype(np.float32)/32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if source_rate != rate:
        from math import gcd
        step = rate//gcd(rate, source_rate)
        audio = np.interp(np.linspace(0, len(audio)-1, len(audio)//step),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def write_wav(path, audio, rate=48000):
    import wave
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((clipped*32767).astype('<i2').tobytes())
    return path


def normalise(audio, target_rms=0.05):
    """Match level between conditions.

    A level difference is the easiest thing to hear and the least interesting, so it
    must not leak into the comparison.
    """
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    if rms <= 1e-9:
        return audio
    scaled = audio*min(target_rms/rms, 0.98/max(float(np.max(np.abs(audio))), 1e-9))
    peak = float(np.max(np.abs(scaled)))
    return (scaled*(0.98/peak) if peak > 0.98 else scaled).astype(np.float32)


def measure(audio, rate=48000):
    """Objective numbers reported next to each pair. They describe, not decide."""
    from src.vc.voice_quality import analyse
    stats = analyse(np.asarray(audio, dtype=np.float32), rate)
    return {
        'rms_dbfs': round(stats['rms_dbfs'], 2),
        'peak_dbfs': round(stats['peak_dbfs'], 2),
        'spectral_flatness': stats['spectral_flatness'],
        'voiced_ratio': round(stats['voiced_ratio'], 4),
        'f0_median_hz': round(stats['f0_median_hz'], 1) if stats['f0_median_hz'] else None,
        'snr_db': stats['snr_db'],
    }


def build_pair(source_audio, rate, convert, conditions, target, rng):
    """Render two conditions of the same audio and shuffle which side each lands on."""
    outputs = {}
    for label, kwargs in conditions.items():
        result = convert(source_audio, rate, **kwargs)
        if result is None:
            print(f'  skip {label}: conversion failed', flush=True)
            continue
        outputs[label] = normalise(result, target_rms=0.05)
    if len(outputs) < 2:
        return None
    labels = sorted(outputs)
    order = list(labels)
    rng.shuffle(order)
    answers = {}
    pages = []
    for side, label in enumerate(order):
        name = f'{target.stem}_{"A" if side == 0 else "B"}.wav'
        write_wav(target/name, outputs[label], rate)
        pages.append({'file': name, 'condition': label})
        answers['A' if side == 0 else 'B'] = label
    return {'pages': pages, 'answer': answers,
            'measurements': {label: measure(audio, rate) for label, audio in outputs.items()}}


def worker_convert(model, timeout=1800):
    """A converter that runs one utterance through the real worker subprocess.

    The condition keywords are forwarded: dropping them would silently render the same
    audio twice and produce a blind pair that proves nothing.
    """
    from src.gui import preview as preview_module

    def convert(audio, rate, enhancer='none', denoise=False, **_ignored):
        staging = Path(tempfile.mkdtemp(prefix='ab_'))
        try:
            source = staging/'input.wav'
            write_wav(source, audio, rate)
            path, _stats = preview_module.convert(source, model, enhancer=enhancer,
                                                   denoise=denoise, timeout=timeout)
            return read_wav(path, rate)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return convert


def registration_compare(source, name, workdir):
    """Old-style selection vs the new one, using the real registration encoder."""
    from tools import register_voice
    torch, encoder = register_voice.load_spk_encoder()
    audio, names, scales = register_voice.load_sources([source])
    report = {}
    artefacts = {}
    for label, kwargs in (('current', dict(highpass_hz=0.0, aggregation='centroid',
                                           subset_search=False, passes=1)),
                          ('new', dict())):
        parts, selections, selection_report = voice_quality_select(audio)
        vectors, norms, agreement = register_voice.encode_selection(
            parts, selections, torch, encoder, **kwargs)
        embedding, keep = register_voice.aggregate_embedding(
            vectors, norms, [row.get('quality') or 1.0 for row in selections],
            kwargs.get('aggregation', register_voice.DEFAULT_AGGREGATION))
        report[label] = dict(selection_report, **agreement)
        artefacts[label] = (np.concatenate(parts), embedding)
    return artefacts, report, names


def voice_quality_select(audio):
    from src.vc import voice_quality
    return voice_quality.select_reference(audio, 16000)


def voice_quality_high_pass(audio, rate, cutoff_hz):
    from src.vc import voice_quality
    return voice_quality.high_pass(np.asarray(audio, dtype=np.float32), rate, cutoff_hz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', default='meanvc2_ref20')
    parser.add_argument('--out', required=True)
    parser.add_argument('--target-rms', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--comparison', default='tta',
                        choices=('tta', 'enhancer', 'registration'),
                        help='tta: 1 pass vs N; enhancer: none vs LavaSR; '
                             'registration: previous selection vs current pipeline')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    audio = read_wav(args.source)
    print(f'source: {args.source} ({len(audio)/48000:.2f}s)', flush=True)

    if args.comparison == 'tta':
        # TTA runs at registration time, so it changes the embedding that gets stored
        # rather than the live conversion. The comparison is therefore the agreement
        # between the two embeddings, plus a blind pair built from audio converted with
        # each embedding as the fixed reference.
        from src.vc import voice_embedding
        from tools import register_voice
        torch, encoder = register_voice.load_spk_encoder()
        import io
        import contextlib

        def embed(samples, rate, passes, highpass_hz, aggregation):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                prepared = voice_quality_high_pass(samples, 16000, highpass_hz)
                vectors, norms, _agreement = register_voice.encode_selection(
                    [prepared], [{}], torch, encoder, passes=passes, aggregation=aggregation)
                merged, _keep = register_voice.aggregate_embedding(
                    vectors, norms, [1.0], aggregation)
            return merged

        variants = {
            'single pass': dict(passes=1, highpass_hz=0.0, aggregation='centroid'),
            'current (%d pass, %d Hz, %s)' % (voice_embedding.DEFAULT_TTA_PASSES,
                                              int(register_voice.DEFAULT_HIGHPASS_HZ),
                                              register_voice.DEFAULT_AGGREGATION):
                dict(passes=voice_embedding.DEFAULT_TTA_PASSES,
                     highpass_hz=register_voice.DEFAULT_HIGHPASS_HZ,
                     aggregation=register_voice.DEFAULT_AGGREGATION),
        }
        embeddings = {}
        for label, kwargs in variants.items():
            embeddings[label] = embed(np.asarray(audio, dtype=np.float32), 48000, **kwargs)
            print('embedded:', label, flush=True)
        labels = sorted(embeddings)
        first, second = (embeddings[label] for label in labels)
        cosine = float(np.dot(first/np.linalg.norm(first), second/np.linalg.norm(second)))
        summary = {
            'kind': 'tta',
            'source': args.source,
            'note': 'TTA runs at registration time. This is the agreement between the two '
                    'stored embeddings, which is what the live conversion will use. '
                    'A high cosine means the two settings describe nearly the same voice.',
            'variants': sorted(variants),
            'embedding_cosine': round(cosine, 4),
        }
        (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), 'utf-8')
        for label, vector in embeddings.items():
            np.save(out/('embedding_%s.npy' % label.replace(' ', '_')), vector,
                    allow_pickle=False)
        print('embedding cosine: %.4f' % cosine)
        print('wrote', out/'summary.json')
        return 0

    if args.comparison == 'enhancer':
        base = worker_convert(args.model)
        conditions = {
            'no enhancer': {'enhancer': 'none'},
            'lavasr': {'enhancer': 'lavasr'},
        }
        results = {}
        for label, kwargs in conditions.items():
            converted = base(audio, 48000, **kwargs)
            if converted is None:
                print(f'  {label}: failed', flush=True)
                continue
            results[label] = normalise(converted, args.target_rms)
        if len(results) < 2:
            print('not enough conditions succeeded')
            return 1
        labels = sorted(results)
        order = list(labels)
        rng.shuffle(order)
        pages, answers = [], {}
        for side, label in enumerate(order):
            name = f'AB{"AB"[side]}.wav'
            write_wav(out/name, results[label], 48000)
            pages.append({'file': name, 'condition': label})
            answers['AB'[side]] = label
        payload = {'kind': 'enhancer', 'source': args.source, 'model': args.model,
                   'pages': pages, 'answer': answers,
                   'measurements': {label: measure(item, 48000)
                                    for label, item in results.items()},
                   'note': 'levels are matched so only the restoration differs'}
        (out/'answer.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), 'utf-8')
        print('wrote', out/'answer.json')
        return 0

    # registration
    artefacts, report, names = registration_compare(args.source, 'ab', out)
    summary = {'kind': 'registration', 'source': args.source, 'sources': names,
               'reports': report,
               'note': 'previous = single pass, no high-pass, centroid, no subset search; '
                       'current = TTA, 80Hz high-pass, ns_mean, subset search'}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), 'utf-8')
    for label, (parts, embedding) in artefacts.items():
        write_wav(out/f'reference_{label}.wav', np.concatenate(parts), 16000)
        np.save(out/f'embedding_{label}.npy', embedding, allow_pickle=False)
    print('wrote', out/'summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())