"""Offline RTF sweep for MeanVC2 buffer and vocoder context.

Run inside the isolated MeanVC2 environment; it never opens an audio device. Each
condition is measured on the same source audio so the numbers are comparable, and
every condition states its own algorithmic delay so a quality/latency trade is visible
rather than implied.
"""
import argparse
import json
import sys
import time
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
        factor = rate//gcd(rate, source_rate)
        audio = np.interp(np.linspace(0, len(audio)-1, len(audio)//factor),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def load_backend(model_folder, threads=1):
    import types
    import importlib.util
    from src.runtime_paths import asset_root
    repo = asset_root()/'vc_models/meanvc2/repo'
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo/'src/infer'))
    package = types.ModuleType('src.model')
    package.__path__ = [str(repo/'src/model')]
    sys.modules['src.model'] = package
    spec = importlib.util.spec_from_file_location('rtf_sweep_upstream', repo/'src/infer/infer_e2e.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, torch


def condition(folder, audio, group_chunks, batch_frames, buffer_ms, steps=None,
              vocoder_context=None):
    """Measure one configuration end to end on the utterance route.

    A temp folder carries the altered runtime so the shipped profile is never edited;
    the backend reads its configuration from runtime.json inside the folder.
    """
    import shutil
    import tempfile
    staging = Path(tempfile.mkdtemp(prefix='rtf_sweep_'))/'profile'
    staging.mkdir(parents=True)
    for name in ('runtime.json', 'reference.wav', 'fixed_embedding.npy'):
        shutil.copy2(folder/name, staging/name)
    runtime = json.loads((staging/'runtime.json').read_text('utf-8'))
    runtime.update(vc_group_chunks=group_chunks, vocoder_batch_frames=batch_frames,
                   model_buffer_ms=buffer_ms)
    if steps is not None:
        runtime['steps'] = steps
    if vocoder_context is not None:
        runtime['vocoder_context'] = vocoder_context
    (staging/'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False), 'utf-8')

    from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
    backend = MeanVC2PhraseBackend(threads=1)
    backend.load(staging)
    backend.select_profile(dict(vc_group_chunks=group_chunks, vocoder_batch_frames=batch_frames,
                                model_buffer_ms=buffer_ms,
                                phrase_repair=True))
    backend.warmup()
    # The backend runs at 16 kHz: 2560 samples per 120 ms chunk, matching convert_utterance.
    from src.vc.resampler import StreamingResampler
    down = StreamingResampler(48000, 16000)
    hop16 = backend.chunk_samples
    hop48 = hop16*(48000//backend.sample_rate)
    padded_length = ((len(audio)+48000+hop48-1)//hop48)*hop48
    padded = np.zeros(padded_length, dtype=np.float32)
    padded[:len(audio)] = audio
    start = time.perf_counter()
    for offset in range(0, len(padded), hop48):
        block = padded[offset:offset+hop48]
        backend.process_chunk(down.process(block))
    elapsed = time.perf_counter()-start
    stats = backend.get_stats()
    chunk_samples = backend.chunk_samples
    backend.unload()
    shutil.rmtree(staging.parent, ignore_errors=True)
    return {
        'elapsed_seconds': round(elapsed, 3),
        'input_seconds': round(len(audio)/48000, 3),
        'rtf': round(elapsed/(len(audio)/48000), 4),
        'algorithmic_buffer_ms': stats.get('algorithmic_buffer_ms'),
        'grid_delay_ms': stats.get('interpolation_grid_delay_ms', 0),
        'phrase_extra_delay_ms': stats.get('phrase_extra_delay_ms', 0),
        'chunk_samples': chunk_samples,
        'steps': stats.get('steps'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', default='meanvc2_ref20')
    parser.add_argument('--out')
    parser.add_argument('--buffers', default='960,1200,1440,1800,2160')
    parser.add_argument('--batches', default='24,36,48')
    parser.add_argument('--groups', default='4,6,8')
    parser.add_argument('--vocoder-contexts', default='36')
    parser.add_argument('--steps-list', default='',
                        help='Comma separated MeanVC2 step counts, e.g. 2,3,4')
    args = parser.parse_args()

    from src.runtime_paths import asset_root
    folder = asset_root()/'models'/args.model
    audio = read_wav(args.source)
    load_backend(folder)  # Warms the vendored import path once, outside the measurements.
    contexts = [int(value) for value in args.vocoder_contexts.split(',')]
    steps_list = [int(value) for value in args.steps_list.split(',')] if args.steps_list else [None]
    results = []
    for group in [int(value) for value in args.groups.split(',')]:
        for batch in [int(value) for value in args.batches.split(',')]:
            for buffer_ms in [int(value) for value in args.buffers.split(',')]:
                for context in contexts:
                    for steps in steps_list:
                        try:
                            row = condition(folder, audio, group, batch, buffer_ms, steps, context)
                        except Exception as error:
                            row = {'error': f'{type(error).__name__}: {error}'}
                        row.update(vc_group_chunks=group, vocoder_batch_frames=batch,
                                   model_buffer_ms=buffer_ms, vocoder_context=context)
                        results.append(row)
                        print(json.dumps(row, ensure_ascii=False), flush=True)
    usable = [row for row in results if 'rtf' in row]
    if not usable:
        print(json.dumps({'error': 'no condition completed', 'results': results},
                         ensure_ascii=False), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps({'results': results}, ensure_ascii=False,
                                                indent=2), 'utf-8')
        return 1
    fastest = min(usable, key=lambda row: row['rtf'])
    summary = {'source': args.source, 'model': args.model, 'fastest': fastest,
               'completed': len(usable), 'conditions': len(results)}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({'summary': summary, 'results': results},
                                            ensure_ascii=False, indent=2), 'utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


if __name__ == '__main__':
    raise SystemExit(main())