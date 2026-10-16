import math
import time

import numpy as np

from src.processors.ai_voice import AIVoiceProcessor
from src.processors.env_passthrough import DelayRing, TransientDetector
from src.nonvoice.offline import splice_raw
from src.vc.bridge import AIBridge
from tests.test_ai_voice import FakeClient, wait

RATE = 48000
BLOCK = 256
RNG = np.random.default_rng(4)


def voiced(seconds, level=.09, f0=140.0):
    t = np.arange(int(seconds * RATE)) / RATE
    wave = np.zeros_like(t)
    for k in range(1, 14):
        wave += (1.0 / k) * np.sin(2 * math.pi * f0 * k * t + k)
    envelope = 0.6 + 0.4 * np.sin(2 * math.pi * 4.0 * t)
    return (level * envelope * wave / 2.0).astype(np.float32)


def with_floor(signal, level=2e-4):
    return (signal + level * RNG.standard_normal(len(signal)).astype(np.float32)).astype(np.float32)


def clap(seconds=.2, level=.5, tau=.1):
    n = int(seconds * RATE)
    attack = RNG.standard_normal(n).astype(np.float32)
    decay = np.exp(-np.arange(n) / (RATE * tau)).astype(np.float32)
    return (level * attack * decay).astype(np.float32)


def hops(signal, size=512):
    return [signal[i:i + size] for i in range(0, len(signal) - size + 1, size)]


def test_delay_ring_reads_by_absolute_index_with_prefill_zeros():
    ring = DelayRing(100)
    ring.write(np.ones(20, dtype=np.float32))
    out = np.zeros(30, dtype=np.float32)
    ring.read_into(out, -10)
    assert not out[:10].any()
    np.testing.assert_array_equal(out[10:], np.ones(20, dtype=np.float32))


def test_delay_ring_wraps_and_stops_at_written():
    ring = DelayRing(64)
    for i in range(10):
        ring.write(np.arange(16, dtype=np.float32) + i * 16)
    out = np.empty(64, dtype=np.float32)
    ring.read_into(out, 96)
    np.testing.assert_array_equal(out, np.arange(96, 160, dtype=np.float32))
    out.fill(7.0)
    ring.read_into(out, 155)
    np.testing.assert_array_equal(out[:5], np.arange(155, 160, dtype=np.float32))
    assert not out[5:].any()
    ring.reset()
    assert ring.written == 0
    out.fill(7.0)
    ring.read_into(out, 0)
    assert not out.any()


def test_transient_detector_quiets_after_the_voice_onset():
    detector = TransientDetector()
    signal = with_floor(voiced(1.5))
    targets = [detector.process(hop) for hop in hops(signal)]
    # The onset itself is a transient; sustained voice must not keep firing.
    assert not any(targets[8:]), 'steady voice kept firing transients'


def test_transient_detector_does_not_repeat_on_sustained_hiss():
    detector = TransientDetector()
    signal = (0.05 * RNG.standard_normal(RATE).astype(np.float32))
    targets = [detector.process(hop) for hop in hops(signal)]
    # The edge itself is a transient, but sustained hiss must not keep re-firing.
    assert targets[0]
    assert not any(targets[10:]), 'sustained hiss kept firing transients'


def test_transient_detector_flags_a_clap_and_quiets_on_voice():
    detector = TransientDetector()
    signal = with_floor(np.concatenate((voiced(.4), clap(.15), voiced(.4))))
    targets = np.array([detector.process(hop) for hop in hops(signal)])
    on_start = int(.4 * RATE / 512)
    assert targets[on_start:on_start + 3].any(), 'clap was not flagged'
    assert not targets[int(.63 * RATE / 512):].any(), 'the voice tail kept firing'


def test_transient_detector_accumulates_blocks_of_any_size():
    detector = TransientDetector()
    clap_signal = with_floor(clap(.12))
    padded = np.concatenate((np.zeros(300, dtype=np.float32), clap_signal))
    targets = [detector.process(padded[i:i + 100]) for i in range(0, len(padded) - 99, 100)]
    assert any(targets)


def test_transient_detector_reset_clears_envelope_state():
    detector = TransientDetector()
    detector.process(with_floor(clap(.15)))
    assert detector._fast > 0.0
    detector.reset()
    assert detector._fast == 0.0 and detector._slow == 0.0 and detector._fill == 0


class FakeNonVoiceGate:
    """Scripted stand-in for src.nonvoice.bridge.NonVoiceBridge."""

    def __init__(self, allow=None):
        self.allow_fn = allow or (lambda base, start, frames, transient: False)
        self.allow_calls = []
        self.loaded = 0
        self.stopped = 0

    def load(self, directory=None):
        self.loaded += 1
        return True

    def stop(self):
        self.stopped += 1

    def mark_epoch(self):
        return 0

    def submit(self, audio):
        pass

    def allow(self, base, start, frames, transient):
        self.allow_calls.append((base, start, frames, transient))
        return self.allow_fn(base, start, frames, transient)


def scripted_clap_gate(low, high):
    """Route a block exactly while its content window overlaps [low, high)."""
    def allow(base, start, frames, transient):
        return start < high and start + frames > low
    return FakeNonVoiceGate(allow)


def passthrough_processor(gate=None):
    bridge = AIBridge(client_factory=FakeClient)
    gate = gate or FakeNonVoiceGate()
    processor = AIVoiceProcessor(bridge=bridge, nonvoice=gate)
    processor.prepare(RATE, BLOCK)
    bridge.status = 'Ready'
    bridge.ack_generation = bridge.generation
    processor._epoch = bridge.generation
    processor._primed = True
    processor._recovering = False
    processor._has_played = True
    return processor, bridge, gate


def simulate(processor, bridge, blocks, count):
    """Feed blocks; the fake worker converts exactly what the model describes.

    Output content for block ``i`` describes the accepted stream one model
    delay earlier, so the ring the processor pulls from and the converted
    content it replaces describe the same moment.
    """
    delay = processor._env_delay
    outs = []
    mixes = []
    for i in range(count):
        start = i * BLOCK - delay
        if start >= 0:
            converted = blocks[start:start + BLOCK] * .5
        else:
            converted = np.zeros(BLOCK, dtype=np.float32)
        assert bridge.output.write(np.asarray(converted, dtype=np.float32))
        source = blocks[i * BLOCK:(i + 1) * BLOCK].copy().reshape(-1, 1)
        out = processor.process(source, RATE)
        outs.append(out[:, 0].copy())
        mixes.append(processor._env_mix)
        assert bridge.input.discard() == BLOCK
    return outs, mixes


def speech_clap_speech():
    speech_a, clap_part, speech_b = 700, 40, 360
    blocks = np.concatenate((with_floor(voiced(speech_a * BLOCK / RATE)),
                             clap(clap_part * BLOCK / RATE),
                             with_floor(voiced(speech_b * BLOCK / RATE))))
    return blocks, speech_a, clap_part, speech_a + clap_part + speech_b


def test_streaming_route_passes_clap_through_at_the_aligned_index():
    blocks, speech_a, clap_part, count = speech_clap_speech()
    gate = scripted_clap_gate(speech_a * BLOCK, (speech_a + clap_part) * BLOCK)
    processor, bridge, gate = passthrough_processor(gate)
    delay = processor._env_delay
    assert delay == round(1490 * 48)
    assert gate.loaded == 1
    outs, mixes = simulate(processor, bridge, blocks, count)
    clap_output = speech_a * BLOCK + delay  # output content that describes the clap
    first = clap_output // BLOCK
    routed = [i for i in range(first, first + clap_part)
              if mixes[i - 1] == 1.0 and mixes[i] == 1.0]
    assert len(routed) >= 10, 'the clap was not routed for a sustained span'
    # On those blocks the output is the raw input, not the model's echo of it.
    for i in routed:
        start = i * BLOCK - delay
        np.testing.assert_allclose(outs[i], blocks[start:start + BLOCK], atol=1e-6)
    # Speech before the clap reached the output converted.
    for i in range(400, 600):
        assert mixes[i] == 0.0
        start = i * BLOCK - delay
        np.testing.assert_allclose(outs[i], blocks[start:start + BLOCK] * .5, atol=1e-6)
    # Once the clap left the window, the route returns to converted output.
    returned = [i for i in range(first + clap_part + 2, count)
                if mixes[i - 1] == 0.0 and mixes[i] == 0.0]
    assert len(returned) >= 20
    for i in returned:
        start = i * BLOCK - delay
        np.testing.assert_allclose(outs[i], blocks[start:start + BLOCK] * .5, atol=1e-6)
    assert mixes[-1] == 0.0
    # The verdict is asked in the epoch's own sample domain.
    assert gate.allow_calls and all(call[0] == 0 for call in gate.allow_calls)


def test_gate_that_never_routes_keeps_converted_output():
    processor, bridge, gate = passthrough_processor()
    assert processor.env_passthrough
    blocks, speech_a, clap_part, count = speech_clap_speech()
    outs, mixes = simulate(processor, bridge, blocks, count)
    assert gate.allow_calls, 'the gate was never consulted'
    first = (speech_a * BLOCK + processor._env_delay) // BLOCK
    for i in range(first, first + clap_part):
        start = i * BLOCK - processor._env_delay
        np.testing.assert_allclose(outs[i], blocks[start:start + BLOCK] * .5, atol=1e-6)
    assert mixes[-1] == 0.0


def test_streaming_route_with_passthrough_off_keeps_converted_output():
    processor, bridge, gate = passthrough_processor(
        FakeNonVoiceGate(lambda base, start, frames, transient: True))
    processor.set_env_passthrough(False)
    delay = processor._env_delay
    blocks, speech_a, clap_part, count = speech_clap_speech()
    outs, mixes = simulate(processor, bridge, blocks, count)
    first = (speech_a * BLOCK + delay) // BLOCK
    for i in range(first, first + clap_part):
        start = i * BLOCK - delay
        np.testing.assert_allclose(outs[i], blocks[start:start + BLOCK] * .5, atol=1e-6)
    assert mixes[-1] == 0.0
    assert processor.bridge.overruns == 0


def test_late_delay_resolution_keeps_the_ring_and_its_alignment():
    gate = scripted_clap_gate(40 * BLOCK, 80 * BLOCK)
    processor, bridge, gate = passthrough_processor(gate)
    ring = processor._env_ring
    bridge.model_stats['model_alignment_delay_ms'] = 2200.0
    blocks = np.concatenate((with_floor(voiced(40 * BLOCK / RATE)),
                             clap(40 * BLOCK / RATE),
                             with_floor(voiced(480 * BLOCK / RATE))))
    outs, mixes = simulate(processor, bridge, blocks, 540)
    assert processor._env_delay == round(2200 * 48)
    assert processor._env_ring is ring, 'a resolved delay must not rebuild the mid-stream ring'
    first = (40 * BLOCK + processor._env_delay) // BLOCK
    routed = [i for i in range(first, first + 40)
              if mixes[i - 1] == 1.0 and mixes[i] == 1.0]
    assert len(routed) >= 10, 'the clap was not routed under the resolved delay'
    for i in routed:
        start = i * BLOCK - processor._env_delay
        np.testing.assert_allclose(outs[i], blocks[start:start + BLOCK], atol=1e-6)


def test_uttering_route_has_no_ring_and_never_routes():
    from src.vc.config import AIParameters
    bridge = AIBridge(AIParameters(delivery='utterance', enhancer='lavasr'), FakeClient)
    gate = FakeNonVoiceGate()
    processor = AIVoiceProcessor(bridge=bridge, nonvoice=gate)
    processor.prepare(RATE, BLOCK)
    assert processor._env_ring is None
    assert not processor._env_usable()
    assert gate.loaded == 0 and gate.stopped >= 1


def test_a_dsp_voice_never_builds_the_ring(monkeypatch):
    bridge = AIBridge(client_factory=FakeClient)
    processor = AIVoiceProcessor(bridge=bridge, nonvoice=FakeNonVoiceGate())
    monkeypatch.setattr('src.processors.ai_voice.is_meanvc2', lambda model: False)
    processor.prepare(RATE, BLOCK)
    assert processor._env_ring is None


def test_processor_stop_stops_the_gate():
    processor, bridge, gate = passthrough_processor()
    processor.stop()
    assert gate.stopped >= 1


def test_foreign_input_trims_shift_the_read_window():
    processor, bridge, gate = passthrough_processor()
    assert bridge.input.write(np.zeros(1024, dtype=np.float32))
    processor._env_accepted = 96000
    bridge.dropped_frames = 512  # the worker trimmed accepted frames it never ran
    processor._env_account_drops()
    assert processor._env_foreign == 512
    position, size = processor._env_drops[0]
    assert size == 512
    assert position == 96000 - bridge.input.available


def test_own_overrun_frames_do_not_shift_the_window():
    processor, bridge, gate = passthrough_processor()
    processor._env_own_pending = 256
    bridge.dropped_frames = 256  # the callback counted its own rejected write
    processor._env_account_drops()
    assert processor._env_foreign == 0
    assert not processor._env_drops


# -- the utterance route: piece-local environment passthrough ----------------

class FakeOfflineGate:
    """Scripted stand-in for src.nonvoice.offline.OfflineGate."""

    def __init__(self, route_spans=(), ambient_emit=None, available=True, error=''):
        self.available = available
        self.error = error
        self._route_spans = list(route_spans)
        self._ambient_emit = ambient_emit
        self.route_lengths = []
        self.ambient_lengths = []

    def route_mask(self, audio):
        self.route_lengths.append(len(audio))
        mask = np.zeros(len(audio), dtype=np.float32)
        for low, high in self._route_spans:
            mask[low:high] = 1.0
        return mask

    def ambient_only(self, audio):
        self.ambient_lengths.append(len(audio))
        if self._ambient_emit is None:
            return np.zeros(0, dtype=np.float32)
        return np.array(self._ambient_emit, dtype=np.float32)


def utterance_bridge(gate=None):
    from src.vc.config import AIParameters
    return AIBridge(AIParameters(delivery='utterance', enhancer='lavasr'), FakeClient, env_gate=gate)


def feed_stream(bridge, samples):
    """Paced writes that never overflow the utterance input ring."""
    size = bridge.chunk_frames
    for start in range(0, len(samples), size):
        while bridge.input.available > bridge.input.capacity - 2 * size:
            time.sleep(.002)
        assert bridge.input.write(samples[start:start + size])
        bridge.input_ready.signal()
        time.sleep(.001)
    # The physical stop: consume the sub-chunk tail and flush the open piece,
    # exactly like the controller's finish button.
    bridge.finish_utterance.set()
    bridge.input_ready.signal()


def drain_output(bridge, seconds=1.0, timeout=60.0):
    """Collect FIFO output; quiet state must hold `seconds` before returning."""
    parts = []
    last_completed = -1
    stable_since = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bridge.output.available:
            component = np.empty(bridge.output.available, dtype=np.float32)
            assert bridge.output.read_into(component)
            parts.append(component)
            stable_since = None
            continue
        completed = int(bridge.utterance_stats.get('completed', 0))
        settled = bridge.utterance_stats.get('state') == '発話待ち' and completed == last_completed
        if not settled:
            last_completed = completed
            stable_since = None
            time.sleep(.002)
            continue
        if stable_since is None:
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= seconds:
            break
        time.sleep(.002)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def two_piece_stream():
    """Preroll, speech A, silent gap, speech B, endpoint silence: 148800."""
    return np.concatenate((np.zeros(19200, dtype=np.float32), voiced(.5, level=.21),
                           np.zeros(43200, dtype=np.float32), voiced(.5, level=.31),
                           np.zeros(38400, dtype=np.float32)))


def piece_parts(level, preroll=9600):
    """The collector's rendering of one 0.5 s speech: preroll+voice+tail.

    Preroll stores the silence the collector saw right after the previous
    piece released it, so piece B (which begins 100 ms after A's endpoint)
    carries 5 preroll frames, but piece A carries all 10.
    """
    return np.concatenate((np.zeros(preroll, dtype=np.float32), voiced(.5, level=level),
                           np.zeros(9600, dtype=np.float32)))


def test_utterance_route_splices_raw_event_audio_into_a_converted_piece():
    gate = FakeOfflineGate(route_spans=[(28000, 32000)])
    bridge = utterance_bridge(gate)
    bridge.set_env_passthrough(True)
    try:
        bridge.load(); bridge.set_active(True)
        wait(lambda: bridge.ack_generation == bridge.generation)
        feed_stream(bridge, np.concatenate((np.zeros(19200, dtype=np.float32), voiced(1.0),
                                            np.zeros(38400, dtype=np.float32))))
        wait(lambda: bridge.utterance_stats['completed'] >= 1)
        piece = np.concatenate((np.zeros(9600, dtype=np.float32), voiced(1.0),
                                np.zeros(9600, dtype=np.float32)))
        assert len(piece) == 67200 and gate.route_lengths == [67200]
        mask = np.zeros(len(piece), dtype=np.float32)
        mask[28000:32000] = 1.0
        np.testing.assert_allclose(drain_output(bridge), splice_raw(piece * .5, piece, mask), atol=1e-6)
        assert abs(bridge.utterance_stats['env_routed_seconds'] - 4000 / 48000) < 1e-9
    finally:
        bridge.stop()


def test_env_gap_filler_emits_events_before_the_speech_they_preceded():
    marker = np.ones(2400, dtype=np.float32) * .9
    gate = FakeOfflineGate(ambient_emit=marker)
    bridge = utterance_bridge(gate)
    bridge.set_env_passthrough(True)
    try:
        bridge.load(); bridge.set_active(True)
        wait(lambda: bridge.ack_generation == bridge.generation)
        stream = two_piece_stream()
        assert len(stream) == 148800
        feed_stream(bridge, stream)
        wait(lambda: bridge.utterance_stats['completed'] >= 2)
        out = drain_output(bridge)
        assert len(out) == 43200 + 2400 + 38400
        np.testing.assert_allclose(out[:43200], piece_parts(.21) * .5, atol=1e-6)
        np.testing.assert_allclose(out[43200:45600], marker)
        np.testing.assert_allclose(out[45600:], piece_parts(.31, preroll=4800) * .5, atol=1e-6)
        assert abs(bridge.utterance_stats['env_filled_seconds'] - 2400 / 48000) < 1e-9
        # The gap between the two pieces: A ends at 52800 and B's preroll picks
        # up 5 silent frames before its onset, so the gap runs to 81600.
        assert gate.ambient_lengths == [28800]
    finally:
        bridge.stop()


def test_env_off_on_the_utterance_route_converts_cleanly():
    gate = FakeOfflineGate(ambient_emit=np.ones(2400, dtype=np.float32) * .9)
    bridge = utterance_bridge(gate)  # bridge default: passthrough off
    try:
        bridge.load(); bridge.set_active(True)
        wait(lambda: bridge.ack_generation == bridge.generation)
        feed_stream(bridge, two_piece_stream())
        wait(lambda: bridge.utterance_stats['completed'] >= 2)
        out = drain_output(bridge)
        np.testing.assert_allclose(out, np.concatenate((piece_parts(.21) * .5,
                                                        piece_parts(.31, preroll=4800) * .5)), atol=1e-6)
        assert gate.route_lengths == [] and gate.ambient_lengths == []
        stats = bridge.utterance_stats
        assert stats['env_routed_seconds'] == 0 and stats['env_filled_seconds'] == 0
    finally:
        bridge.stop()


def test_unavailable_gate_fails_closed_and_reports_once():
    gate = FakeOfflineGate(available=False, error='テスト用: モデル不在')
    bridge = utterance_bridge(gate)
    bridge.set_env_passthrough(True)
    try:
        bridge.load(); bridge.set_active(True)
        wait(lambda: bridge.ack_generation == bridge.generation)
        feed_stream(bridge, two_piece_stream())
        wait(lambda: bridge.utterance_stats['completed'] >= 2)
        out = drain_output(bridge)
        np.testing.assert_allclose(out, np.concatenate((piece_parts(.21) * .5,
                                                        piece_parts(.31, preroll=4800) * .5)), atol=1e-6)
        stats = bridge.utterance_stats
        assert stats['env_error'] == 'テスト用: モデル不在'
        # The never-converted gap was withheld, not played.
        assert abs(stats['env_fill_dropped_seconds'] - 28800 / 48000) < 1e-9
    finally:
        bridge.stop()
