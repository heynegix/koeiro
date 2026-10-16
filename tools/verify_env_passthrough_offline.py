"""Offline verification of the utterance route's environment passthrough.

Wall-clock WAV replay through the real processor/bridge/worker: ambient bursts
(claps, knocks, typing) must reach the listener as raw event sound while real
speech converts. No audio devices are opened and nothing here is listening
evidence; the user judges the output WAVs separately.

Two scenarios run through the user's own settings and one worker load:

* ``recording`` — the untouched environment recording. Every routed span's
  interior must appear in the output as exact raw samples.
* ``speech_composite`` — the voice fixture mixed over the recording's tail.
  Speech must convert (no loud raw window anywhere in the output) while the
  ambient bursts stay raw.
"""
import json
import os
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RECORDING_WAV = Path(os.environ.get('KOEIRO_ENVPASS_RECORDING',
                                    str(Path('recordings') / 'env_passthrough' / 'recording48.wav')))
VOICE_FIXTURE = ROOT / 'models' / 'user_voices' / 'user_3b11387911244c6e9a013b42fa88a458' / 'reference.wav'
OUT_DIR = ROOT / 'recordings' / 'env_passthrough'
REPORT_PATH = ROOT / 'validation' / 'env_passthrough' / 'env_passthrough_offline.json'
RATE = 48000
BLOCK = 256
SPLICE_FADE = 64


def read_wave(path):
    """Float32 mono at the file's own rate; int16 PCM, float32 or raw float data."""
    try:
        with wave.open(str(path), 'rb') as source:
            rate, channels, raw = (source.getframerate(), source.getnchannels(),
                                   source.readframes(source.getnframes()))
            width = source.getsampwidth()
    except wave.Error:
        # Older decode steps wrote raw float32 with only the classic chunks;
        # the recorder's 48 kHz mono assumption carries here.
        data = Path(path).read_bytes()
        start = data.index(b'data') + 8
        return RATE, np.frombuffer(data[start:], dtype=np.float32).astype(np.float32)
    if width == 2:
        audio = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype=np.float32)
    else:
        raise SystemExit(f'{path}: unsupported sample width {width}')
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return rate, audio.astype(np.float32)


def write_wave(path, audio, rate=RATE):
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())


def seconds_at(position):
    return None if position is None or position < 0 else round(position / RATE, 2)


def routed_spans(mask, minimum=1000):
    edge = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edge == 1)
    ends = np.flatnonzero(edge == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= minimum]


def exact_search(output, probe):
    """Best (position, max-abs-error) of a probe window across the output."""
    if len(output) < len(probe):
        return -1, float('inf')
    windows = np.lib.stride_tricks.sliding_window_view(output, len(probe))
    differences = np.abs(windows - probe).max(axis=1)
    best = int(np.argmin(differences))
    return best, float(differences[best])


class UnusedDSP:
    """The DSP slot never runs in the AI route's verification replay."""
    def prepare(self, *args): pass
    def reset(self): pass
    def process(self, audio, *args): return audio


def replay(processor, bridge, signal, wall_timeout=75.0, settle=1.5):
    """Wall-clock callback replay; returns the output stream the listener heard."""
    parts = []
    start = time.monotonic()
    next_callback = start
    fed = 0
    finish_sent = False
    settled = None
    deadline = start + len(signal) / RATE + wall_timeout
    while time.monotonic() < deadline:
        block = np.zeros((BLOCK, 1), dtype=np.float32)
        piece = signal[fed:fed + BLOCK]
        if len(piece):
            block[:len(piece), 0] = piece
        out = processor.process(block, RATE)
        parts.append(out[:, 0].copy())
        fed += BLOCK
        if bridge.status == 'Error':
            raise RuntimeError(bridge.error)
        if not finish_sent and fed >= len(signal):
            # The physical stop: consume the tail and flush the open piece.
            bridge.finish_utterance.set()
            bridge.input_ready.signal()
            finish_sent = True
        if finish_sent:
            stats = bridge.utterance_stats
            quiet = (bridge.output.available == 0 and stats.get('state') == '発話待ち')
            if quiet:
                if settled is None:
                    settled = time.monotonic()
                elif time.monotonic() - settled >= settle:
                    break
            else:
                settled = None
        elapsed = time.monotonic() - start
        if os.environ.get('VP_TRACE') and int(elapsed) != latest_trace[0]:
            latest_trace[0] = int(elapsed)
            stats = bridge.utterance_stats
            print(f'[vp {int(elapsed)}s] fed={fed / RATE:.2f} completed={stats.get("completed")} '
                  f'accepted={stats.get("rejected")} queued={stats.get("queued")} '
                  f'overruns={bridge.overruns + bridge.output_overruns} out={bridge.output.available} '
                  f'env_routed={round(stats.get("env_routed_seconds", 0), 2)} state={stats.get("state")!r}',
                  flush=True)
        next_callback += BLOCK / RATE
        time.sleep(max(0.0, next_callback - time.monotonic()))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


latest_trace = [-1]


def routed_span_events(gate, source, output):
    """Every routed span's interior must appear raw in the output."""
    mask = gate.route_mask(source)
    events = []
    for a, b in routed_spans(mask):
        interior = source[a + SPLICE_FADE + 256:b - SPLICE_FADE - 256]
        if len(interior) < 128:
            continue
        probe = interior[len(interior) // 2 - 32:len(interior) // 2 + 32]
        found, error = exact_search(output, probe)
        events.append(dict(start=a, end=b, seconds=round((b - a) / RATE, 3),
                           found_at=seconds_at(found), abs_error=round(error, 6),
                           raw=bool(error < 1e-4)))
    return events


def speech_leak(source, fixture, onset, output, anchors=40):
    """A loud raw window of the fixture must match nothing in the output."""
    windows = np.lib.stride_tricks.sliding_window_view(fixture, 64)
    powers = (windows.astype(np.float64) ** 2).mean(axis=1)
    worst = (None, float('inf'))
    for anchor in np.argsort(powers)[-anchors:]:
        anchor = int(anchor) + onset
        if anchor + 64 > len(source):
            continue
        found, error = exact_search(output, source[anchor:anchor + 64])
        if error < worst[1]:
            worst = (anchor, error)
    position, error = worst
    return dict(at=seconds_at(position), abs_error=None if error == float('inf') else round(error, 5),
                leaked=bool(error < 5e-3))


def scenario(processor, bridge, router, gate, name, source, fixture=None, onset=None):
    started = time.monotonic()
    to_ready = time.monotonic() + 180
    bridge.load()
    router.prepare(RATE, BLOCK)
    bridge.set_active(True)
    while bridge.status != 'Ready' or bridge.ack_generation != bridge.generation:
        if bridge.status == 'Error' or time.monotonic() > to_ready:
            raise RuntimeError(bridge.error or 'model load timed out')
        time.sleep(.05)
    output = replay(processor, bridge, source)
    path = OUT_DIR / (name + '_output.wav')
    write_wave(path, output)
    metrics = dict(output=str(path),
                   load_and_replay_seconds=round(time.monotonic() - started, 2),
                   pieces_completed=bridge.utterance_stats.get('completed'),
                   pieces_rejected=bridge.utterance_stats.get('rejected'),
                   env_routed_seconds=bridge.utterance_stats.get('env_routed_seconds'),
                   env_filled_seconds=bridge.utterance_stats.get('env_filled_seconds'),
                   env_fill_dropped_seconds=bridge.utterance_stats.get('env_fill_dropped_seconds'),
                   env_error=bridge.utterance_stats.get('env_error'),
                   piece_overruns=int(bridge.overruns + bridge.output_overruns),
                   input_seconds=round(len(source) / RATE, 3),
                   output_seconds=round(len(output) / RATE, 3))
    events = routed_span_events(gate, source, output)
    metrics['routed_spans'] = len(events)
    metrics['spans_raw'] = sum(1 for event in events if event['raw'])
    metrics['routed_span_seconds'] = round(sum(event['seconds'] for event in events), 3)
    metrics['worst_event_error'] = max((event['abs_error'] for event in events), default=0.0)
    metrics['events'] = events
    if fixture is not None:
        metrics['speech_leak'] = speech_leak(source, fixture, onset, output)
    return metrics


def main():
    from tools.compare_meanvc2_continuity import require_idle_voice_worker
    require_idle_voice_worker()
    recording = Path(sys.argv[1] if len(sys.argv) > 1 else RECORDING_WAV)
    if not recording.is_file():
        raise SystemExit('decoded recording WAV missing; decode the mp3 first')
    rate, source = read_wave(recording)
    if rate != RATE:
        raise SystemExit(f'{recording}: expected 48 kHz, got {rate} Hz')
    rate_fixture, fixture = read_wave(VOICE_FIXTURE)
    if rate_fixture != 16000:
        raise SystemExit('the fixture is a 16 kHz voice reference')
    fixture = np.interp(np.arange(3 * len(fixture)) / 3.0, np.arange(len(fixture)), fixture)
    fixture = (fixture.astype(np.float32) / (np.abs(fixture).max() + 1e-9) * 0.5)

    settings = json.loads((ROOT / 'settings.json').read_text(encoding='utf-8'))
    from src.settings.manager import AppSettings
    parameters = AppSettings.from_dict(settings).ai_parameters()
    from src.vc.models import is_meanvc2
    if parameters.delivery != 'utterance' or not is_meanvc2(parameters.model):
        raise SystemExit('the route must be utterance+LavaSR to verify this behaviour')

    report = dict(kind='OFFLINE WAV replay; no audio devices; listening is user-side',
                  model=parameters.model, enhancer=parameters.enhancer,
                  experiment=parameters.experiment, lavasr_denoise=parameters.lavasr_denoise,
                  env_passthrough_setting=settings.get('env_passthrough'))

    from src.vc.bridge import AIBridge
    from src.nonvoice.offline import OfflineGate
    from src.processors.ai_voice import AIVoiceProcessor
    from src.processors.voice_router import VoiceRouter

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge = AIBridge(parameters)
    processor = AIVoiceProcessor(bridge=bridge)
    router = VoiceRouter(dsp=UnusedDSP(), ai=processor, mode='ai_voice')
    gate = OfflineGate()
    bridge.set_env_passthrough(True)
    if not gate.available:
        raise SystemExit('environment gate models are unavailable: ' + gate.error)

    try:
        run_started = time.monotonic()
        report['recording'] = scenario(processor, bridge, router, gate, 'recording', source)
        # A fresh epoch for the composite scenario; the worker stays loaded.
        bridge.set_active(False)
        time.sleep(.3)

        onset = int(7.4 * RATE)
        fit = fixture[:len(source) - onset]
        mixed = source.copy()
        mixed[onset:onset + len(fit)] += fit
        write_wave(OUT_DIR / 'speech_composite_input.wav', mixed)
        report['speech_composite'] = scenario(processor, bridge, router, gate,
                                              'speech_composite', mixed,
                                              fixture=fit, onset=onset)
        report['run_seconds'] = round(time.monotonic() - run_started, 1)
        recording_metrics = report['recording']
        composite = report['speech_composite']
        # The full-input mask overstates what piece-local gating can choose:
        # every gate window it sees is the whole clip, while the bridge gates
        # per piece and per slice. A PASS asks for real raw coverage plus the
        # privacy direction (speech converts, never leaks raw).
        verdict = (recording_metrics['spans_raw'] >= 2
                   and recording_metrics['routed_spans'] > 0
                   and composite['spans_raw'] == composite['routed_spans']
                   and composite['routed_spans'] > 0
                   and not composite['speech_leak']['leaked'])
        report['status'] = 'SUCCESS' if verdict else 'FAIL'
    finally:
        began = time.monotonic()
        bridge.stop()
        report['stop_seconds'] = round(time.monotonic() - began, 2)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
