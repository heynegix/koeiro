import numpy as np
import pytest

from src.audio.meters import amplitude_to_db, peak_level
from src.processors.base import AudioProcessor
from src.processors.chain import ProcessorChain, SafetyClampProcessor
from src.processors.gain import GainProcessor
from src.processors.noise_gate import NoiseGateProcessor
from src.processors.passthrough import PassthroughProcessor


@pytest.mark.parametrize("frames", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_passthrough_preserves_buffer(frames, rate):
    source = np.linspace(-1, 1, frames, dtype=np.float32).reshape(-1, 1)
    original = source.copy()
    assert PassthroughProcessor().process(source, rate) is source
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("db", [-20, -6, 0, 6, 20])
def test_gain(db):
    processor = GainProcessor(db)
    processor.prepare(48000, 256)
    audio = np.full((256, 1), 0.05, dtype=np.float32)
    assert processor.process(audio, 48000) is audio
    np.testing.assert_allclose(audio, 0.05 * 10 ** (db / 20), rtol=1e-6)


def test_gain_live_change_is_smooth_and_converges():
    processor = GainProcessor()
    processor.prepare(48000, 256)
    processor.set_gain(20)
    audio = np.ones((256, 1), dtype=np.float32)
    processor.process(audio, 48000)
    assert 1 < audio[0, 0] < 1.1
    assert np.all(np.diff(audio[:, 0]) >= 0)
    previous = audio[-1, 0]
    audio.fill(1)
    processor.process(audio, 48000)
    assert 0 <= audio[0, 0] - previous < 0.05
    for _ in range(40):
        audio.fill(1)
        processor.process(audio, 48000)
    np.testing.assert_allclose(audio, 10, rtol=1e-5)


@pytest.mark.parametrize("value", [-21, 21, float("nan"), float("inf")])
def test_invalid_gain(value):
    with pytest.raises(ValueError):
        GainProcessor(value)


def process_constant(processor, level, blocks=1, rate=48000, frames=256):
    audio = np.empty((frames, 1), dtype=np.float32)
    for _ in range(blocks):
        audio.fill(level)
        processor.process(audio, rate)
    return audio


def test_gate_silence_and_below_threshold():
    gate = NoiseGateProcessor(-45)
    gate.prepare(48000, 256)
    np.testing.assert_array_equal(process_constant(gate, 0.0001, 20), 0)
    np.testing.assert_array_equal(process_constant(gate, 0, 20), 0)


def test_gate_attack_hold_release_and_reset():
    gate = NoiseGateProcessor(-45)
    gate.prepare(48000, 256)
    first = process_constant(gate, 0.1)
    assert 0 < first[0, 0] < first[-1, 0] < 0.1
    opened = process_constant(gate, 0.1, 40)
    np.testing.assert_allclose(opened, 0.1, atol=1e-6)
    holding = process_constant(gate, 0.0001, 2)
    np.testing.assert_allclose(holding, 0.0001, atol=1e-7)
    released = process_constant(gate, 0.0001, 180)
    assert np.max(released) < 1e-8
    gate.reset()
    np.testing.assert_array_equal(process_constant(gate, 0.0001), 0)


def test_gate_threshold_change_and_hysteresis():
    gate = NoiseGateProcessor(-45)
    gate.prepare(48000, 256)
    process_constant(gate, 0.1, 40)
    # -46 dB is below open but above close threshold, so gate stays open.
    audio = process_constant(gate, 10 ** (-46 / 20), 80)
    assert np.min(audio) > 0.004
    gate.set_threshold(-10)
    audio = process_constant(gate, 0.02, 200)
    assert np.max(audio) < 1e-7


@pytest.mark.parametrize("value", [-81, -9, float("nan"), float("inf")])
def test_invalid_gate(value):
    with pytest.raises(ValueError):
        NoiseGateProcessor(value)


@pytest.mark.parametrize("frames", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_chain_format_and_clamp(frames, rate):
    chain = ProcessorChain(gain_db=20)
    chain.prepare(rate, frames)
    for _ in range(200):
        audio = np.full((frames, 1), 0.8, dtype=np.float32)
        result = chain.process(audio, rate)
    assert result is audio
    assert result.shape == (frames, 1) and result.dtype == np.float32
    np.testing.assert_array_equal(result, 1)


class RecordingMain(AudioProcessor):
    def prepare(self, sample_rate, max_frames):
        self.prepared = sample_rate, max_frames

    def process(self, audio, sample_rate):
        self.seen = float(audio[-1, 0])
        np.multiply(audio, 2, out=audio)
        return audio


def test_chain_order_and_replaceable_main():
    main = RecordingMain()
    chain = ProcessorChain(main_processor=main, gain_db=6)
    chain.prepare(48000, 256)
    audio = process_constant(chain, 0.1, 80)
    assert main.prepared == (48000, 256)
    assert main.seen == pytest.approx(0.1 * 10 ** (6 / 20), rel=1e-5)
    np.testing.assert_allclose(audio, 2 * main.seen, rtol=1e-5)


def test_clamp_removes_nonfinite_and_clips():
    clamp = SafetyClampProcessor()
    clamp.prepare(48000, 6)
    audio = np.array([np.nan, np.inf, -np.inf, 3, -3, 0.5], dtype=np.float32).reshape(-1, 1)
    clamp.process(audio, 48000)
    np.testing.assert_array_equal(audio[:, 0], [0, 0, 0, 1, -1, 0.5])


def test_meter():
    assert peak_level(np.array([[-0.8], [0.2]], dtype=np.float32)) == pytest.approx(0.8)
    assert amplitude_to_db(1) == 0
    assert amplitude_to_db(0.1) == pytest.approx(-20)
    assert amplitude_to_db(0) == -80
