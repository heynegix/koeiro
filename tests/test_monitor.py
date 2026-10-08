import numpy as np
import pytest

from src.audio.monitor import MonitorPlayer, MonitorTap
from .fakes import FakeBackend, OUTPUT


def test_tap_roundtrip_and_stats():
    tap = MonitorTap(seconds=1.0, rate=48000)
    tap.active = True
    block = (np.arange(480, dtype=np.float32) / 480)
    tap.offer(block)
    out = np.zeros(480, dtype=np.float32)
    assert tap.read_into(out) == 480
    np.testing.assert_array_equal(out, block)
    assert tap.offered == 480 and tap.dropped == 0


def test_tap_drops_oldest_never_blocks():
    tap = MonitorTap(seconds=0.1, rate=48000)
    tap.active = True
    tap.offer(np.ones(4800, dtype=np.float32))
    tap.offer(np.full(4800, 2.0, dtype=np.float32))
    assert tap.dropped > 0
    out = np.zeros(4800, dtype=np.float32)
    assert tap.read_into(out) == 4800
    assert float(out.mean()) == pytest.approx(2.0)


def test_tap_read_empty_gives_zero_frames():
    tap = MonitorTap()
    tap.active = True
    assert tap.read_into(np.zeros(256, dtype=np.float32)) == 0
    tap.active = False
    tap.offer(np.ones(256, dtype=np.float32))
    assert tap.read_into(np.zeros(256, dtype=np.float32)) == 0


def test_player_plays_offered_audio_and_counts_gaps():
    backend = FakeBackend()
    tap = MonitorTap()
    player = MonitorPlayer(backend)
    player.start(OUTPUT, 48000, 256, tap)
    assert player.running
    baseline_zeroed = player.zeroed
    tap.offer(np.full(256, 0.5, dtype=np.float32))
    stream = backend.streams[-1]
    stream.tick()
    np.testing.assert_allclose(stream.outdata[:, 0], 0.5, rtol=1e-5)
    np.testing.assert_allclose(stream.outdata[:, 1], 0.5, rtol=1e-5)
    assert player.played == 256 and player.zeroed == baseline_zeroed
    stream.tick()
    assert player.zeroed == baseline_zeroed + 256
    assert (stream.outdata == 0.0).all()
    player.stop()
    assert not player.running


def test_player_start_failure_is_a_clean_error():
    from src.audio.engine import AudioEngineError
    backend = FakeBackend()
    backend.fail_start = True
    player = MonitorPlayer(backend)
    with pytest.raises(AudioEngineError):
        player.start(OUTPUT, 48000, 256, MonitorTap())
    assert not player.running


def test_engine_monitor_lifecycle_alongside_main_stream():
    from src.audio.engine import AudioEngine, EngineConfig
    from tests.fakes import INPUT
    backend = FakeBackend()
    engine = AudioEngine(backend=backend)
    engine.start(EngineConfig(INPUT, OUTPUT, monitor=True, monitor_device=OUTPUT))
    try:
        assert engine.monitor_player is not None and engine.monitor_player.running
        assert engine.monitor_error == ""
        engine.monitor_tap.offer(np.full(256, 0.25, dtype=np.float32))
        monitor_stream = backend.streams[-1]
        monitor_stream.tick()
        np.testing.assert_allclose(monitor_stream.outdata[:, 0], 0.25, rtol=1e-5)
    finally:
        engine.stop()
    assert engine.monitor_player is None


def test_engine_starts_and_stops_the_monitor_while_running():
    """The monitor is applied live: no Stop/Start cycle for device or level."""
    from src.audio.engine import AudioEngine, EngineConfig
    from tests.fakes import INPUT
    backend = FakeBackend()
    engine = AudioEngine(backend=backend)
    engine.start(EngineConfig(INPUT, OUTPUT))
    try:
        assert engine.monitor_player is None and engine.monitor_tap.active is False
        assert engine.start_monitor(OUTPUT) is True
        assert engine.monitor_player.running and engine.monitor_error == ""
        engine.set_monitor_gain(-6.0)
        assert engine.monitor_player.gain == pytest.approx(10**(-6.0/20.0))
        engine.stop_monitor()
        assert engine.monitor_player is None and engine.monitor_tap.active is False
        # A restarted monitor keeps the level and plays the tap again.
        assert engine.start_monitor(OUTPUT) is True
        engine.monitor_tap.offer(np.full(256, 0.25, dtype=np.float32))
        stream = backend.streams[-1]
        stream.tick()
        np.testing.assert_allclose(stream.outdata[:, 0], 0.25*10**(-6.0/20.0), rtol=1e-4)
    finally:
        engine.stop()
    assert engine.monitor_player is None


def test_monitor_gain_is_bounded_and_ignores_invalid_values():
    from src.audio.engine import AudioEngine
    engine = AudioEngine(backend=FakeBackend())
    engine.set_monitor_gain(-20.0)
    assert engine.monitor_gain_db == -20.0
    engine.set_monitor_gain(999.0)
    assert engine.monitor_gain_db == 12.0
    engine.set_monitor_gain('loud')
    assert engine.monitor_gain_db == 12.0
    engine.set_monitor_gain(float('nan'))
    assert engine.monitor_gain_db == 12.0


def test_test_tone_is_bounded_and_faded():
    from src.audio.monitor import test_tone
    tone = test_tone(48000, 0.2, 0.3)
    assert len(tone) == 9600 and np.isfinite(tone).all()
    assert float(np.max(np.abs(tone))) <= 0.3 + 1e-6
    # Both edges are faded, so the chime cannot click.
    assert abs(float(tone[0])) < 1e-3 and abs(float(tone[-1])) < 1e-3
    assert float(np.max(np.abs(tone[len(tone)//4:3*len(tone)//4]))) > 0.2


def test_play_test_tone_uses_the_monitor_stream_and_cleans_up():
    from src.audio.monitor import play_test_tone
    backend = FakeBackend()
    label = play_test_tone(backend, OUTPUT, 48000, 0.05, -6.0, timeout=0.05)
    assert label == OUTPUT.label
    stream = backend.streams[-1]
    assert stream.kwargs['device'] == OUTPUT.index
    assert stream.kwargs['channels'] == 2 and stream.kwargs['blocksize'] == 256
    assert stream.closed and not stream.active


def test_play_test_tone_reports_a_device_that_cannot_open():
    from src.audio.engine import AudioEngineError
    from src.audio.monitor import play_test_tone
    backend = FakeBackend()
    backend.fail_start = True
    with pytest.raises(AudioEngineError):
        play_test_tone(backend, OUTPUT, 48000, 0.05, -6.0, timeout=0.05)
