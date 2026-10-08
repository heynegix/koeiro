import gc
import queue
from types import SimpleNamespace

from src.audio.diagnostics import AudioDiagnostics
from src.audio.performance import TimingStats
from src.utils.logging import BoundedLogHandler


def test_p95_tracks_tail_not_single_outlier():
    stats = TimingStats()
    for elapsed in [1_000_000]*94 + [2_000_000]*5 + [30_000_000]:
        stats.record(elapsed, 256, 48000)
    result = stats.snapshot()
    assert 2 <= result.p95_ms <= 2.01
    assert result.maximum_ms == 30
    assert result.deadline_exceeded == 1
    stats.reset()
    assert stats.snapshot().p95_ms == 0


def test_histogram_is_bounded_for_extreme_stall():
    stats = TimingStats()
    stats.record(1_000_000_000, 256, 48000)
    assert stats.snapshot().p95_ms == 1000
    assert len(stats._histogram) == 14901


def test_gc_observer_does_not_leak_across_restarts():
    diagnostics = AudioDiagnostics()
    before = len(gc.callbacks)
    for _ in range(10):
        diagnostics.attach()
        diagnostics.attach()
        assert len(gc.callbacks) == before+1
        gc.collect()
        diagnostics.detach()
        diagnostics.detach()
    assert len(gc.callbacks) == before
    assert diagnostics.gc_count >= 10


def test_shared_wasapi_bursts_are_not_false_late_arrivals():
    diagnostics = AudioDiagnostics()
    diagnostics.entered(1_000_000_000, None, 5.33)
    gap, host, late = diagnostics.entered(1_010_000_000, SimpleNamespace(currentTime=1), 5.33)
    assert gap == 10 and not late
    assert diagnostics.entered(1_060_000_000, None, 5.33)[2]


def test_event_ring_preserves_only_latest_events():
    diagnostics = AudioDiagnostics()
    for i in range(200):
        diagnostics.completed(i*1_000_000, 30, 30, True, 1, 5, SimpleNamespace(_level=0.5))
    snapshot = diagnostics.snapshot()
    assert snapshot['event_count'] == 200
    assert len(snapshot['events']) == 128
    assert snapshot['events'][0][0] == .072
    assert snapshot['events'][-1][0] == .199


def test_log_overload_drops_without_waiting():
    pending = queue.Queue(maxsize=1)
    handler = BoundedLogHandler(pending)
    handler.enqueue('one')
    handler.enqueue('two')
    assert handler.dropped == 1
    assert pending.get_nowait() == 'one'


def test_failed_stream_start_detaches_gc_observer():
    import pytest
    from src.audio.engine import AudioEngine, AudioEngineError, EngineConfig
    from tests.fakes import FakeBackend, INPUT, OUTPUT
    backend = FakeBackend()
    backend.fail_start = True
    engine = AudioEngine(backend=backend)
    before = len(gc.callbacks)
    with pytest.raises(AudioEngineError):
        engine.start(EngineConfig(INPUT, OUTPUT))
    assert not engine.diagnostics.attached
    assert len(gc.callbacks) == before


def test_native_timing_counts_only_processing_calls():
    import numpy as np
    from src.processors.female_dsp import FemaleDSPProcessor
    from src.presets.female_presets import load_preset
    processor = FemaleDSPProcessor(load_preset('Anime Test'))
    processor.prepare(48000, 256)
    assert processor._native.timing_snapshot()['count'] == 0
    for _ in range(12):
        processor.process(np.full((256, 1), .1, dtype=np.float32), 48000)
    snapshot = processor._native.timing_snapshot()
    assert snapshot['count'] == 12
    assert 0 < snapshot['average_ms'] <= snapshot['maximum_ms']
