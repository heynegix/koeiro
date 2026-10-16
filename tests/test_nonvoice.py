import time
import wave
from pathlib import Path

import numpy as np
import pytest

from src.nonvoice import models as model_files
from src.nonvoice.bridge import (
    B_THRESHOLD, DecisionStore, Decimator, MARGIN_HI, NonVoiceBridge,
    RATE48, SILERO_FRAME, VOCAL_THRESHOLD, YAM_HOP, YAM_WINDOW,
)
from src.nonvoice.offline import OfflineGate, eroded_flags, span_allowed
from src.processors.env_passthrough import TransientDetector
from src.vc.bridge import AIBridge
from src.vc.config import AIParameters
from tests.test_ai_voice import FakeClient, wait
from tests.test_env_passthrough import drain_output, feed_stream

try:
    import onnxruntime  # noqa: F401
    HAVE_ONNXRUNTIME = True
except Exception:
    HAVE_ONNXRUNTIME = False

VOICE_FIXTURE = (Path(__file__).resolve().parents[1] / 'models' / 'user_voices' /
                 'user_3b11387911244c6e9a013b42fa88a458' / 'reference.wav')
needs_models = pytest.mark.skipif(
    model_files.models_dir() is None or not HAVE_ONNXRUNTIME,
    reason='nonvoice models or onnxruntime not installed')


# -- model files ------------------------------------------------------------

@needs_models
def test_event_names_all_exist_in_the_class_map():
    directory = model_files.models_dir()
    names = set(model_files.class_map_names(directory / model_files.CLASS_MAP_FILE).values())
    assert not (model_files.EVENT_NAMES - names)
    assert not (model_files.VOCAL_NAMES - names)
    assert not (model_files.EVENT_NAMES & model_files.VOCAL_NAMES)


@needs_models
def test_event_ids_are_sorted_and_cover_clapping():
    ids = model_files.event_ids(model_files.models_dir() / model_files.CLASS_MAP_FILE)
    assert len(ids) >= len(model_files.EVENT_NAMES) - 2
    assert np.array_equal(ids, np.sort(ids))
    assert 58 in ids  # Clapping


def test_models_dir_is_none_when_only_some_files_exist(tmp_path):
    (tmp_path / 'models' / 'nonvoice').mkdir(parents=True)
    (tmp_path / 'models' / 'nonvoice' / model_files.SILERO_FILE).write_bytes(b'x')
    original_asset = model_files.asset_root
    model_files.asset_root = lambda: tmp_path
    try:
        assert model_files.models_dir() is None
    finally:
        model_files.asset_root = original_asset


# -- decimator --------------------------------------------------------------

def test_decimator_preserves_a_band_limited_tone():
    seconds = 0.5
    t = np.arange(int(seconds * RATE48)) / RATE48
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    decimator = Decimator()
    output = np.concatenate([decimator.process(tone[i:i + 4800])
                             for i in range(0, len(tone), 4800)])
    assert len(output) == len(tone) // 3
    body = output[400:-400]
    assert abs(float(np.sqrt(np.mean(body ** 2))) - 0.5 / np.sqrt(2)) < 0.01
    spectrum = np.abs(np.fft.rfft(body))
    peak = float(np.argmax(spectrum)) * 16000.0 / len(body)
    assert abs(peak - 1000.0) < 10.0


def test_decimator_is_chunk_size_independent_for_multiples_of_three():
    signal = np.random.default_rng(3).standard_normal(48000).astype(np.float32)
    one = Decimator()
    a = np.concatenate([one.process(signal[i:i + 4800]) for i in range(0, len(signal), 4800)])
    two = Decimator()
    b = np.concatenate([two.process(signal[i:i + 1440]) for i in range(0, len(signal), 1440)])
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_decimator_reset_clears_the_carry():
    decimator = Decimator()
    decimator.process(np.ones(4800, dtype=np.float32))
    decimator.reset()
    output = decimator.process(np.zeros(4800, dtype=np.float32))
    assert not output.any()


# -- decision store ---------------------------------------------------------

LO = SILERO_FRAME * 100          # span frame 100..100
HI = LO + 240
FIRST_VETO_FRAME = LO // SILERO_FRAME - 3           # 97
LAST_VETO_FRAME = (HI + 3072) // SILERO_FRAME + 2   # 108


def store_with(high_speech=(), high_events=(), frames=220, events=None, high_vocals=()):
    store = DecisionStore()
    for frame in range(frames):
        store.publish_speech(frame, 0.9 if frame in high_speech else 0.0)
    for window in range(events if events is not None else 40):
        store.publish_event(window, 0.9 if window in high_events else 0.0)
        store.publish_vocal(window, 0.9 if window in high_vocals else 0.0)
    return store


def test_a_strong_event_routes_without_a_transient():
    store = store_with(high_events=(15,))
    assert store.allow_span(LO, HI, False)


def test_a_weak_event_needs_a_transient():
    store = store_with(high_events=(15,))
    store.publish_event(15, 0.10)
    assert not store.allow_span(LO, HI, False)
    assert store.allow_span(LO, HI, True)


def test_a_vocal_sound_vetoes_routing():
    """A cough-like window must never route, whatever the event score says."""
    store = store_with(high_events=(15,), high_vocals=(15,))
    assert not store.allow_span(LO, HI, False)
    assert not store.allow_span(LO, HI, True)
    # Just under the threshold still routes; the cutoff is the constant.
    store.publish_vocal(15, VOCAL_THRESHOLD - 0.001)
    assert store.allow_span(LO, HI, False)
    # Only covering windows veto, same reach as event windows.
    assert store_with(high_events=(15,), high_vocals=(13,)).allow_span(LO, HI, False)
    assert not store_with(high_events=(15,), high_vocals=(15,)).allow_span(LO, HI, False)


def test_below_b_threshold_never_routes():
    store = store_with()
    store.publish_event(15, B_THRESHOLD - 0.001)
    assert not store.allow_span(LO, HI, False)
    assert not store.allow_span(LO, HI, True)


def test_eroded_speech_vetoes_even_a_strong_event():
    store = store_with(high_speech=(97, 98, 99), high_events=(15,))
    assert not store.allow_span(LO, HI, False)
    assert not store.allow_span(LO, HI, True)


def test_an_isolated_high_speech_frame_does_not_veto():
    store = store_with(high_speech=(98,), high_events=(15,))
    assert store.allow_span(LO, HI, False)


def test_veto_window_covers_96ms_before_and_192ms_after():
    first = FIRST_VETO_FRAME
    last = LAST_VETO_FRAME
    assert store_with(high_events=(15,),
                      high_speech=(first - 2, first - 1, first)).allow_span(LO, HI, False)
    assert not store_with(high_events=(15,),
                          high_speech=(first - 1, first, first + 1)).allow_span(LO, HI, False)
    assert not store_with(high_events=(15,),
                          high_speech=(last - 1, last, last + 1)).allow_span(LO, HI, False)
    assert store_with(high_events=(15,),
                      high_speech=(last, last + 1, last + 2)).allow_span(LO, HI, False)


def test_missing_speech_frames_count_as_speech():
    store = DecisionStore()
    for window in range(40):
        store.publish_event(window, 0.9)
    assert not store.allow_span(LO, HI, False)


def test_missing_event_windows_contribute_nothing():
    store = store_with(frames=220)
    assert not store.allow_span(LO, HI, True)


def test_event_windows_cover_up_to_96ms_before_the_span():
    window_first = -(-(LO - YAM_WINDOW) // YAM_HOP)
    assert store_with(high_events=(window_first,)).allow_span(LO, HI, False)
    assert not store_with(high_events=(window_first - 1,)).allow_span(LO, HI, False)


def test_event_windows_stop_at_the_span_end():
    window_last = HI // YAM_HOP
    assert store_with(high_events=(window_last,)).allow_span(LO, HI, False)
    assert not store_with(high_events=(window_last + 1,)).allow_span(LO, HI, False)


def test_aged_out_speech_frames_veto():
    store = DecisionStore()
    for frame in range(1075):
        store.publish_speech(frame, 0.0)
    for window in range(40):
        store.publish_event(window, 0.9)
    assert not store.allow_span(SILERO_FRAME * 20, SILERO_FRAME * 20 + 240, False)
    assert store.allow_span(LO, HI, False)


# -- bridge lifecycle without models ----------------------------------------

def test_bridge_is_inert_before_load():
    bridge = NonVoiceBridge()
    assert bridge.status == 'off'
    assert bridge.mark_epoch() == 0
    bridge.submit(np.zeros(4800, dtype=np.float32))
    assert not bridge.allow(0, 0, 4800, True)
    bridge.stop()
    bridge.stop()


def test_bridge_reports_unavailable_without_model_files(monkeypatch):
    monkeypatch.setattr(model_files, 'models_dir', lambda: None)
    bridge = NonVoiceBridge()
    assert bridge.load() is False
    assert bridge.status == 'unavailable'
    assert bridge.error
    bridge.submit(np.zeros(4800, dtype=np.float32))
    assert not bridge.allow(0, 0, 4800, True)
    bridge.stop()


def test_bridge_reports_error_when_models_fail_to_load(tmp_path):
    (tmp_path / model_files.SILERO_FILE).write_bytes(b'not a model')
    bridge = NonVoiceBridge()
    assert bridge.load(tmp_path) is True
    deadline = time.monotonic() + 30.0
    while bridge.status == 'loading' and time.monotonic() < deadline:
        time.sleep(0.02)
    assert bridge.status == 'error'
    assert not bridge.allow(0, 0, 4800, True)
    bridge.stop()


# -- the real models --------------------------------------------------------

def wait_until(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('timed out waiting for the analysis thread')


def feed(bridge, signal, chunk=4800):
    """Submit at producer pace: never let the 2 s input ring overflow."""
    for i in range(0, len(signal), chunk):
        bridge.submit(signal[i:i + chunk])
        deadline = time.monotonic() + 5.0
        while bridge.pending > 4 * chunk and time.monotonic() < deadline:
            time.sleep(0.005)


def upsample3(signal16):
    index = np.arange(3 * len(signal16)) / 3.0
    return np.interp(index, np.arange(len(signal16)), signal16).astype(np.float32)


def read_wav16(path):
    with wave.open(str(path)) as source:
        raw = np.frombuffer(source.readframes(source.getnframes()), np.int16)
        if source.getnchannels() == 2:
            raw = raw.reshape(-1, 2).mean(axis=1)
    return raw.astype(np.float32) / 32768.0


def clap48(rng, seconds=0.12, level=0.5):
    n = int(seconds * RATE48)
    noise = rng.standard_normal(n).astype(np.float32)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / RATE48)
    spectrum[(freqs < 200.0) | (freqs > 6000.0)] = 0
    shaped = np.fft.irfft(spectrum, n).astype(np.float32)
    t = np.arange(n) / RATE48
    wave_out = level * shaped / (np.abs(shaped).max() + 1e-9)
    wave_out *= np.clip(t / 0.002, 0.0, 1.0) * np.exp(-t / 0.05)
    return wave_out.astype(np.float32)


@needs_models
@pytest.mark.skipif(not VOICE_FIXTURE.is_file(), reason='speech fixture missing')
def test_real_gate_routes_a_clap_and_rejects_real_speech():
    bridge = NonVoiceBridge()
    try:
        assert bridge.load()
        wait_until(lambda: bridge.status in ('ready', 'error'))
        assert bridge.status == 'ready'

        speech48 = upsample3(read_wav16(VOICE_FIXTURE))
        feed(bridge, speech48)
        wait_until(lambda: bridge.speech_frames >= 105)

        # Real speech must never route, even when a transient flag is claimed.
        for start in range(int(0.5 * RATE48), int(2.8 * RATE48), int(0.2 * RATE48)):
            assert not bridge.allow(0, start, 4800, True), f'route at {start / RATE48:.1f}s'

        # A clap burst in silence routes through the transient path.
        rng = np.random.default_rng(7)
        signal = np.zeros(int(2.2 * RATE48), dtype=np.float32)
        burst = clap48(rng)
        at = int(1.0 * RATE48)
        signal[at:at + len(burst)] += burst
        base = bridge.mark_epoch()
        feed(bridge, signal)
        wait_until(lambda: bridge.speech_frames >= 176 and bridge.event_windows >= 30)

        detector = TransientDetector()
        transient = detector.process(signal[at:at + 4800])
        assert transient, 'the production detector must flag the clap'
        assert bridge.allow(base, at, 4800, transient), 'the clap was not routed'
    finally:
        bridge.stop()


# -- the offline piece-local gate --------------------------------------------

def test_span_allowed_matches_decision_store_on_shared_rings():
    """The piece-local mirror must reproduce every streamed verdict exactly."""
    rng = np.random.default_rng(11)
    frames, windows = 300, 40
    probs = np.where(rng.random(frames) < .3, .5 + .5 * rng.random(frames),
                     .5 * rng.random(frames)).astype(np.float32)
    events = rng.random(windows).astype(np.float32)
    vocals = np.where(rng.random(windows) < .35, .3 + .7 * rng.random(windows),
                      .3 * rng.random(windows)).astype(np.float32)
    store = DecisionStore()
    for frame, probability in enumerate(probs):
        store.publish_speech(frame, float(probability))
    for window, score in enumerate(events):
        store.publish_event(window, float(score))
        store.publish_vocal(window, float(vocals[window]))
    flags = eroded_flags(probs)
    checked = 0
    for low in range(4 * SILERO_FRAME, frames * SILERO_FRAME, 19 * SILERO_FRAME):
        for length in (240, 512, 1536, 4800, 9600):
            high = low + length
            first = max(0, low // SILERO_FRAME - 3)
            last = (high + MARGIN_HI) // SILERO_FRAME + 2
            if last >= frames or high // YAM_HOP >= windows:
                continue  # the buffers would fall short of the margin/window need
            for transient in (False, True):
                # Interior buffers: both sides hold every frame they consult.
                assert span_allowed(flags, events, vocals, low, high, transient) \
                    == store.allow_span(low, high, transient), (low, length, transient)
                checked += 1
    assert checked >= 100


def test_piece_head_preroll_never_vetoes_and_missing_tails_always_veto():
    flags = eroded_flags(np.zeros(100, dtype=np.float32))
    events = np.ones(30, dtype=np.float32)
    vocals = np.zeros(30, dtype=np.float32)
    # Ahead of frame 0 are the piece's own subthreshold preroll frames, not
    # unwritten speech: an event right at the head routes.
    assert span_allowed(flags, events, vocals, 0, 240, True)
    # The margin past the last computed frame is unwritten speech: veto.
    assert not span_allowed(flags, events, vocals, 96 * SILERO_FRAME,
                            96 * SILERO_FRAME + 240, True)


def test_a_veto_frame_at_the_first_index_of_the_window_still_vetoes():
    """Speech ending exactly at the veto window's first frame must veto.

    A window whose only eroded frame sits at relative index 0 once slipped
    through (`np.flatnonzero(...).any()` reads [0] as falsy) and could route
    raw speech; this pins the direct check.
    """
    flags = np.zeros(100, dtype=bool)
    low = 50 * SILERO_FRAME
    first = max(0, low // SILERO_FRAME - 3)
    flags[first] = True
    events = np.ones(30, dtype=np.float32)
    vocals = np.zeros(30, dtype=np.float32)
    assert not span_allowed(flags, events, vocals, low, low + 240, True)
    flags[first] = False
    assert span_allowed(flags, events, vocals, low, low + 240, True)


def test_span_allowed_vetoes_on_a_covering_vocal_window():
    flags = eroded_flags(np.zeros(100, dtype=np.float32))
    events = np.ones(30, dtype=np.float32)
    vocals = np.zeros(30, dtype=np.float32)
    low = 5 * YAM_HOP
    assert span_allowed(flags, events, vocals, low, low + 240, True)
    vocals[5] = VOCAL_THRESHOLD + 0.01
    assert not span_allowed(flags, events, vocals, low, low + 240, True)
    vocals[5] = VOCAL_THRESHOLD - 0.01
    assert span_allowed(flags, events, vocals, low, low + 240, True)


@needs_models
@pytest.mark.skipif(not VOICE_FIXTURE.is_file(), reason='speech fixture missing')
def test_offline_gate_routes_a_clap_piece_and_rejects_a_speech_piece():
    gate = OfflineGate()
    assert gate.available
    speech = upsample3(read_wav16(VOICE_FIXTURE))[:int(3 * RATE48)]
    assert not gate.route_mask(speech).any(), 'speech leaked through the offline mask'

    rng = np.random.default_rng(7)
    signal = np.zeros(int(2.6 * RATE48), dtype=np.float32)
    burst = clap48(rng, seconds=.26)
    at = int(1.0 * RATE48)
    signal[at:at + len(burst)] += burst
    mask = gate.route_mask(signal)
    routed = int(mask.sum())
    assert routed, 'the clap was not routed'
    # One A-class window covers ~0.96 s, so silence inside that window routes
    # along; what must not happen is audio beyond the window's span.
    assert routed < 2.5 * RATE48, 'silence beyond the burst window was routed too'
    centre = at + len(burst) // 2
    assert mask[centre:centre + 512].all(), 'the clap interior was not routed'


@needs_models
def test_offline_gate_skips_quiet_content_without_losing_events():
    gate = OfflineGate()
    assert gate.available
    silence = np.full(RATE48, 1e-5, dtype=np.float32)
    assert not gate.route_mask(silence).any()
    # Below the skip peak even a shaped burst routes nothing: it could not have
    # reached the B threshold anyway, and the models were never run for it.
    rng = np.random.default_rng(7)
    burst = clap48(rng, seconds=.26)
    quiet = np.zeros(int(2.6 * RATE48), dtype=np.float32)
    at = int(1.0 * RATE48)
    quiet[at:at + len(burst)] += burst / (np.abs(burst).max() + 1e-9) * 0.01
    assert float(np.abs(quiet).max()) < 0.02
    assert not gate.route_mask(quiet).any()


def nearest_window(out, probe):
    """Position and error of the output window closest to a raw probe."""
    windows = np.lib.stride_tricks.sliding_window_view(out, len(probe))
    differences = np.abs(windows - probe).max(axis=1)
    best = int(np.argmin(differences))
    return best, float(differences[best])


@needs_models
@pytest.mark.skipif(not VOICE_FIXTURE.is_file(), reason='speech fixture missing')
def test_real_utterance_route_splices_a_clap_and_never_raw_speech():
    """End to end on the bridge: the real gate decides, FakeClient converts."""
    bridge = AIBridge(AIParameters(delivery='utterance', enhancer='lavasr'), FakeClient)
    bridge.set_env_passthrough(True)
    try:
        bridge.load(); bridge.set_active(True)
        wait(lambda: bridge.ack_generation == bridge.generation)
        # The clap piece: its interior reaches the listener bit exact.
        rng = np.random.default_rng(7)
        burst = clap48(rng, seconds=.26)
        at = int(1.0 * RATE48)
        clap_stream = np.zeros(int(2.6 * RATE48), dtype=np.float32)
        clap_stream[at:at + len(burst)] += burst
        feed_stream(bridge, clap_stream)
        wait_until(lambda: bridge.utterance_stats['completed'] >= 1)
        out = drain_output(bridge)
        probe = clap_stream[at + RATE48 // 16:at + RATE48 // 16 + 64]
        position, difference = nearest_window(out, probe)
        assert difference < 1e-5, 'the clap interior was not preserved raw'

        # The speech piece: no loud raw window may appear anywhere in output.
        speech = upsample3(read_wav16(VOICE_FIXTURE))
        feed_stream(bridge, speech)
        wait_until(lambda: bridge.utterance_stats['completed'] >= 2)
        out = drain_output(bridge)
        probes = np.lib.stride_tricks.sliding_window_view(speech, 64)
        powers = (probes.astype(np.float64) ** 2).mean(axis=1)
        for anchor in np.argsort(powers)[-5:]:
            _, difference = nearest_window(out, speech[anchor:anchor + 64])
            assert difference > 1e-3, f'raw speech leaked at {anchor/RATE48:.2f}s'
        assert not bridge.utterance_stats['env_error']
    finally:
        bridge.stop()
