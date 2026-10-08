from dataclasses import replace
import time

import numpy as np
import pytest

from src.audio.controller import AudioController
from src.audio.devices import choose_device, enumerate_devices
from src.audio.engine import AudioEngine, AudioEngineError, EngineConfig
from src.processors.base import AudioProcessor
from src.processors.chain import ProcessorChain
from .fakes import FakeBackend, INPUT, OUTPUT, flags


@pytest.fixture
def setup_engine():
    backend = FakeBackend()
    engine = AudioEngine(backend=backend)
    config = EngineConfig(INPUT, OUTPUT)
    yield engine, backend, config
    backend.fail_close = False
    engine.stop()


def test_device_direction_filter_and_saved_fallback():
    devices = enumerate_devices(FakeBackend())
    assert choose_device(devices, "input") == INPUT
    assert choose_device(devices, "output", preferred_api=INPUT.host_api) == OUTPUT
    assert choose_device(devices, "input", {"name": "removed", "host_api": INPUT.host_api}) == INPUT
    assert choose_device([], "input") is None


@pytest.mark.parametrize("frames", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_start_stop_ten_cycles(setup_engine, frames, rate):
    engine, backend, config = setup_engine
    config = replace(config, buffer_size=frames, sample_rate=rate)
    for _ in range(10):
        engine.start(config)
        stream = backend.streams[-1]
        for _ in range(40):
            stream.tick()
        engine.poll()
        np.testing.assert_allclose(stream.outdata, 0.1, atol=1e-4)
        assert stream.outdata.dtype == np.float32 and np.isfinite(stream.outdata).all()
        assert engine.reported_latency_ms == 20
        assert config.buffer_latency_ms == pytest.approx(1000 * frames / rate)
        engine.stop()
        engine.stop()
        assert not engine.running and stream.closed
    assert len(backend.streams) == 10


def test_no_duplicate_stream(setup_engine):
    engine, backend, config = setup_engine
    engine.start(config)
    with pytest.raises(AudioEngineError, match="already running"):
        engine.start(config)
    assert len(backend.streams) == 1 and backend.streams[0].active


def test_start_failure_cleanup_then_retry(setup_engine):
    engine, backend, config = setup_engine
    backend.fail_start = True
    with pytest.raises(AudioEngineError):
        engine.start(config)
    assert not engine.running and backend.streams[0].closed
    backend.fail_start = False
    engine.start(config)
    assert engine.running


def test_abort_failure_still_closes(setup_engine):
    engine, backend, config = setup_engine
    engine.start(config)
    backend.fail_abort = True
    engine.stop()
    assert not engine.running and backend.streams[0].closed


def test_close_failure_prevents_reopen(setup_engine):
    engine, backend, config = setup_engine
    engine.start(config)
    backend.fail_close = True
    with pytest.raises(AudioEngineError):
        engine.stop()
    with pytest.raises(AudioEngineError, match="already running"):
        engine.start(config)
    assert len(backend.streams) == 1


def test_host_api_mismatch_rejected_before_stream(setup_engine):
    engine, backend, config = setup_engine
    with pytest.raises(AudioEngineError):
        engine.start(replace(config, output_device=replace(OUTPUT, host_api="MME")))
    assert not backend.streams


def test_device_index_changed_before_start(setup_engine):
    engine, backend, config = setup_engine
    backend.removed = True
    with pytest.raises(AudioEngineError, match="disconnected"):
        engine.start(config)
    assert not backend.streams


@pytest.mark.parametrize("mode", ["inactive", "finished", "stalled"])
def test_device_disconnect_watchdog(setup_engine, mode):
    engine, backend, config = setup_engine
    engine.start(config)
    stream = backend.streams[0]
    if mode == "inactive":
        stream.active = False
    elif mode == "finished":
        stream.kwargs["finished_callback"]()
    else:
        engine._last_callback -= 3
    with pytest.raises(AudioEngineError, match="disconnected"):
        engine.poll()
    assert stream.closed and not engine.running


def test_xruns_count_without_callback_logging(setup_engine, caplog):
    engine, backend, config = setup_engine
    engine.start(config)
    caplog.clear()
    for _ in range(100):
        backend.streams[0].tick(flags(input_underflow=True, output_overflow=True))
    assert engine.underflows == 100 and engine.overflows == 100
    assert not caplog.records
    engine._last_log_time -= 2
    engine.poll()
    assert "Underflow=100 Overflow=100" in caplog.text
    assert engine.running


def test_channel_fallback_downmix_and_mono_output(setup_engine):
    engine, backend, config = setup_engine
    backend.stereo_input_only = True
    backend.mono_output_only = True
    engine.start(config)
    stream = backend.streams[0]
    assert stream.kwargs["channels"] == (2, 1)
    stream.indata[:, 0] = 0.2
    stream.indata[:, 1] = 0.4
    for _ in range(80):
        stream.tick()
    np.testing.assert_allclose(stream.outdata, 0.3, atol=1e-5)


def test_oversized_callback_silences_and_aborts(setup_engine):
    engine, backend, config = setup_engine
    engine.start(config)
    output = np.ones((512, 2), dtype=np.float32)
    with pytest.raises(FakeBackend.CallbackAbort):
        engine._callback(np.ones((512, 1), dtype=np.float32), output, 512, None, flags())
    np.testing.assert_array_equal(output, 0)
    with pytest.raises(AudioEngineError, match="処理エラー"):
        engine.poll()


class BrokenMain(AudioProcessor):
    def process(self, audio, sample_rate):
        raise RuntimeError("Processor failed")


def test_processor_failure_silences_and_releases():
    backend = FakeBackend()
    engine = AudioEngine(chain=ProcessorChain(main_processor=BrokenMain()), backend=backend)
    with pytest.raises(AudioEngineError):
        engine.start(EngineConfig(INPUT, OUTPUT))
    np.testing.assert_array_equal(backend.streams[0].outdata, 0)
    assert backend.streams[0].closed and not engine.running


def test_device_close_failure_still_stops_main_worker():
    class OwnedWorker(AudioProcessor):
        stopped=False
        def process(self,audio,sample_rate):return audio
        def stop(self):self.stopped=True
    main=OwnedWorker()
    backend=FakeBackend()
    engine=AudioEngine(ProcessorChain(main),backend)
    engine.start(EngineConfig(INPUT,OUTPUT))
    backend.fail_close=True
    with pytest.raises(AudioEngineError):engine.stop()
    assert main.stopped and engine.running  # uncertain stream remains protected
    backend.fail_close=False
    engine.stop()


def test_nonfinite_input_is_sanitized(setup_engine):
    engine, backend, config = setup_engine
    engine.start(config)
    backend.streams[0].indata[0, 0] = np.nan
    backend.streams[0].tick()
    assert np.isfinite(backend.streams[0].outdata).all()


def test_thirty_minutes_of_frames_with_reused_buffers(setup_engine):
    """Accelerated 30 min of sample frames; this is not a wall-clock soak."""
    engine, backend, config = setup_engine
    engine.start(config)
    stream = backend.streams[0]
    work = engine._work
    gate_workspace = engine.chain.gate._envelope
    gain_workspace = engine.chain.gain._ramp
    blocks = (30 * 60 * config.sample_rate) // config.buffer_size
    for block in range(blocks):
        if block % 10000 == 0:
            # Change parameters during the run, including gate close/reopen.
            engine.chain.gain.set_gain(-6 if block % 20000 else 6)
            engine.chain.gate.set_threshold(-10 if block % 20000 else -45)
        stream.tick()
    assert blocks == 337500
    assert engine._work is work
    assert engine.chain.gate._envelope is gate_workspace
    assert engine.chain.gain._ramp is gain_workspace
    assert np.isfinite(stream.outdata).all()
    assert np.max(np.abs(stream.outdata)) <= 1
    assert engine.underflows == engine.overflows == 0
    engine.poll()


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Controller timed out")


def test_controller_serializes_ten_cycles_and_shutdown():
    backend = FakeBackend()
    controller = AudioController(AudioEngine(backend=backend))
    try:
        wait_for(lambda: controller.devices_revision == 1)
        for _ in range(10):
            request = controller.start(EngineConfig(INPUT, OUTPUT))
            wait_for(lambda: controller.snapshot.request_id == request and controller.snapshot.state == "Running")
            request = controller.stop()
            wait_for(lambda: controller.snapshot.request_id == request and controller.snapshot.state == "Stopped")
        assert len(backend.streams) == 10
        assert all(stream.closed for stream in backend.streams)
    finally:
        controller.shutdown()
        wait_for(lambda: not controller.alive)


def test_controller_device_error_is_recoverable():
    backend = FakeBackend()
    controller = AudioController(AudioEngine(backend=backend))
    try:
        wait_for(lambda: controller.devices_revision == 1)
        controller.start(EngineConfig(INPUT, OUTPUT))
        wait_for(lambda: controller.snapshot.state == "Running")
        backend.streams[0].active = False
        wait_for(lambda: controller.snapshot.state == "Error")
        assert not controller.engine.running
        controller.start(EngineConfig(INPUT, OUTPUT))
        wait_for(lambda: controller.snapshot.state == "Running")
    finally:
        controller.shutdown()
        wait_for(lambda: not controller.alive)
