"""Whole-file conversion through the production utterance path, per device.

`tools/benchmark_meanvc2_realtime.py` measures the streaming cadence, one 160ms
hop at a time. This measures the route the 一括変換 delivery uses: the recording
is already finished, so `src/vc/utterance.py` hands the whole padded input to the
backend and trims its reported delay afterwards. Both branches of that function
are timed on the same audio, and the two outputs are compared, so the report says
what the whole-file pass bought and what it changed.

Offline replay only: no audio device is opened and nothing is listened to.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from src.vc.meanvc2 import MeanVC2Backend
from src.vc.models import DEFAULT_VOICE_ID, VOICE_PROFILES, profile, is_meanvc2
from src.vc.resampler import StreamingResampler
from src.vc.utterance import _convert_core


class _HopLoopOnly:
    """The same backend with the whole-file branch hidden, for comparison."""

    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        if name == 'convert_offline':
            # Forwarding it would make both branches the whole-file one, which
            # is exactly the comparison that looks identical and proves nothing.
            raise AttributeError(name)
        return getattr(self.backend, name)


def read_48k(path, seconds):
    audio, rate = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample_poly(audio, 48000, rate).astype(np.float32)
    wanted = round(seconds * 48000)
    if len(audio) < wanted:
        audio = np.tile(audio, int(np.ceil(wanted / len(audio))))
    return audio[:wanted]


def compare(left, right):
    difference = np.abs(left.astype('float64') - right.astype('float64'))
    signal = max(float(np.sqrt(np.mean(left.astype('float64')**2))), 1e-12)
    rms = float(np.sqrt(np.mean(difference**2)))
    return dict(samples=int(len(left)), identical=bool(np.array_equal(left, right)),
                max_abs_diff=float(difference.max()), rms_diff=rms,
                diff_to_signal_db=20*math.log10(max(rms, 1e-12)/signal))


def measure(device, args, audio):
    import psutil
    import torch
    backend = MeanVC2Backend(threads=args.threads, device=device)
    backend.load(ROOT/'models'/profile(args.model)['folder'])
    backend.select_profile(profile(args.model))
    backend.warmup()
    stats = backend.get_stats()
    delay = round((stats['algorithmic_buffer_ms'] + stats.get('interpolation_grid_delay_ms', 0)
                   + stats.get('phrase_extra_delay_ms', 0))*48)
    hop = backend.chunk_samples*(48000//backend.sample_rate)
    audio_seconds = len(audio)/48000
    hops = ((len(audio)+delay+48000+hop-1)//hop)*hop//hop

    stage: dict = {}
    started = time.perf_counter()
    backend.offline_replay_enabled = False
    whole = _convert_core(backend, StreamingResampler(48000, 16000), StreamingResampler(16000, 48000),
                          audio, delay, hop)
    whole_seconds = time.perf_counter()-started
    stage['offline_stage_seconds'] = backend.get_stats().get('offline_stage_seconds')

    # The buffered route, with the stage captures built before the clock starts:
    # capture is a one-off per shape and would otherwise be charged to the audio.
    capture_seconds = None
    if backend.offline_stages is not None:
        backend.offline_replay_enabled = True
        # Two groups at the shipped six-chunk grouping: the first builds the
        # cache-less stages, the second the 24-frame-cache ones, so both are
        # captured before the timed run. One group would capture only the
        # former and charge the latter's capture to `buffered_seconds`.
        prefix = resample_poly(audio, 16000, 48000)[:backend.chunk_samples*12].astype(np.float32)
        started = time.perf_counter()
        backend.convert_offline(np.ascontiguousarray(prefix), replay=True)
        capture_seconds = time.perf_counter()-started
        started = time.perf_counter()
        buffered = _convert_core(backend, StreamingResampler(48000, 16000), StreamingResampler(16000, 48000),
                                 audio, delay, hop)
        buffered_seconds = time.perf_counter()-started
        replay_stats = backend.get_stats()['offline_replay']
        buffered_stages = backend.get_stats().get('offline_stage_seconds')
    else:
        buffered, buffered_seconds, replay_stats, buffered_stages = None, None, None, None

    loop_backend = _HopLoopOnly(backend)
    per_hop_stages = []
    original = backend.process_chunk

    def timed(audio_chunk):
        result = original(audio_chunk)
        per_hop_stages.append(backend.stats['last_stage_ms'])
        return result

    backend.process_chunk = timed
    started = time.perf_counter()
    looped = _convert_core(loop_backend, StreamingResampler(48000, 16000), StreamingResampler(16000, 48000),
                           audio, delay, hop)
    hop_seconds = time.perf_counter()-started
    backend.process_chunk = original

    record = dict(stats, device=str(device), torch=str(torch.__version__),
                  cuda_available=bool(torch.cuda.is_available()),
                  cuda_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                  tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                  audio_seconds=audio_seconds, delay_samples=delay, hops=hops,
                  whole_seconds=whole_seconds, whole_rtf=whole_seconds/audio_seconds,
                  whole_chunk_equivalent_ms=whole_seconds/hops*1000,
                  hop_loop_seconds=hop_seconds, hop_loop_rtf=hop_seconds/audio_seconds,
                  speedup_over_hop_loop=hop_seconds/whole_seconds,
                  offline_stage_seconds=stage['offline_stage_seconds'],
                  buffered_seconds=buffered_seconds,
                  buffered_rtf=(buffered_seconds/audio_seconds if buffered_seconds else None),
                  buffered_speedup=(whole_seconds/buffered_seconds if buffered_seconds else None),
                  buffered_capture_seconds=capture_seconds,
                  buffered_offline_stage_seconds=buffered_stages,
                  replay_stats=replay_stats,
                  buffered_comparison=(None if buffered is None else compare(whole, buffered)),
                  hop_stage_mean_ms={key: float(np.mean([entry[key] for entry in per_hop_stages]))
                                     for key in ('features', 'vc', 'vocoder')},
                  process_ram_bytes=psutil.Process().memory_info().rss,
                  output_rms=float(np.sqrt(np.mean(whole.astype('float64')**2))),
                  output_finite=bool(np.isfinite(whole).all()),
                  comparison=compare(looped, whole))
    backend.unload()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return whole, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--source', type=Path, default=None, help='WAV to convert; default: the voice reference')
    parser.add_argument('--model', choices=[key for key in VOICE_PROFILES if is_meanvc2(key)],
                        default=DEFAULT_VOICE_ID)
    parser.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=['cpu'])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    folder = ROOT/'models'/profile(args.model)['folder']
    source = args.source or folder/'reference.wav'
    audio = read_48k(source, args.seconds)
    print(f'input {len(audio)} samples at 48 kHz ({len(audio)/48000:.2f} s) from {source}', flush=True)

    routes, results, errors = {}, {}, {}
    for device in args.devices:
        print(f'--- {device} ---', flush=True)
        try:
            results[device], routes[device] = measure(device, args, audio)
        except Exception as error:
            errors[device] = f'{type(error).__name__}: {error}'
            routes[device] = dict(device=str(device), error=errors[device])
        print(json.dumps(routes[device], indent=2, ensure_ascii=False), flush=True)

    report = dict(stage='complete', profile=args.model, source=str(source),
                  input_kind='Offline whole-file conversion; no audio devices',
                  platform=sys.platform, python=sys.version.split()[0],
                  routes=[routes[key] for key in args.devices])
    if len(args.devices) > 1 and len(results) == len(args.devices):
        first, second = args.devices[0], args.devices[1]
        report['comparison'] = dict(baseline=first, against=second,
                                    **compare(results[first], results[second]))
    if len(args.devices) == 1:
        report.update(routes[args.devices[0]])
    for device, out in results.items():
        suffix = '.wav' if len(args.devices) == 1 else f'.{device}.wav'
        sf.write(args.report.with_suffix(suffix), out, 48000, subtype='FLOAT')
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if 'comparison' in report:
        print(json.dumps(report['comparison'], indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit('Failed: ' + ', '.join(f'{key}: {value}' for key, value in errors.items()))


if __name__ == '__main__':
    main()
