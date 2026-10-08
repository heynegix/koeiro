from dataclasses import asdict, replace

import numpy as np
import pytest

from src.audio.engine import AudioEngine, EngineConfig
from src.processors.chain import ProcessorChain
from src.processors.dsp_parameters import DSPParameters
from src.processors.female_dsp import FemaleDSPProcessor
from src.processors.formant import formant_factor
from src.processors.pitch import pitch_factor
from src.presets.female_presets import PRESETS, load_preset
from src.settings.manager import AppSettings, SettingsManager
from .fakes import FakeBackend, FakeStream, INPUT, OUTPUT


def render(processor, signal, rate=48000, block=256):
    processor.prepare(rate, block)
    result = np.empty_like(signal)
    for offset in range(0, len(signal), block):
        audio = signal[offset:offset+block].reshape(-1, 1).copy()
        output = processor.process(audio, rate)
        assert output is audio
        result[offset:offset+len(audio)] = output[:, 0]
    return result


def dominant_frequency(signal, rate=48000):
    signal = signal[rate//2:]
    spectrum = np.abs(np.fft.rfft(signal*np.hanning(len(signal))))
    return np.argmax(spectrum)*rate/len(signal)


@pytest.mark.parametrize("block", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_female_bypass_is_exact(block, rate):
    audio = np.random.default_rng(1).uniform(-0.8, 0.8, block*8).astype(np.float32)
    processor = FemaleDSPProcessor()
    result = render(processor, audio, rate, block)
    np.testing.assert_array_equal(result, audio)
    assert processor.algorithmic_latency_samples == 0


@pytest.mark.parametrize("field,values", [
    ("pitch", [-12.1, 12.1, np.nan, np.inf, "3", True]),
    ("formant", [-6.1, 6.1, np.nan, np.inf, None]),
    ("brightness", [-1, 101, np.nan, "60", True]),
    ("wet", [-0.1, 1.1, np.nan]),
    ("mode", ["AI", None]), ("quality", ["unknown"]),
    ("low_cut", [1]), ("limiter", ["ON"]),
])
def test_invalid_parameters(field, values):
    for value in values:
        with pytest.raises(ValueError):
            replace(DSPParameters(), **{field: value})


@pytest.mark.parametrize("semitones", [-12, -3, 0, 3, 12])
@pytest.mark.parametrize("quality", ["low_latency", "balanced"])
def test_pitch_changes_frequency_without_duration_change(semitones, quality):
    rate = 48000
    t = np.arange(rate*2)/rate
    source = (0.2*np.sin(2*np.pi*220*t)).astype(np.float32)
    p = DSPParameters(mode="female_dsp", pitch=semitones, formant=0,
                      brightness=50, low_cut=False, quality=quality)
    processor = FemaleDSPProcessor(p)
    result = render(processor, source)
    assert result.dtype == np.float32 and result.shape == source.shape
    assert np.isfinite(result).all()
    # Short windows trade spectral accuracy for latency; measure in cents
    # rather than accepting a fixed error that is excessive at low pitches.
    cents = 1200*np.log2(dominant_frequency(result)/(220*pitch_factor(semitones)))
    assert abs(cents) <= (35 if quality == "low_latency" else 25)


def vowel(rate=48000, seconds=2, f0=120):
    t = np.arange(round(rate*seconds))/rate
    audio = np.zeros(len(t))
    for harmonic in range(1, 60):
        f = harmonic*f0
        amplitude = (0.02/harmonic + np.exp(-0.5*((f-720)/120)**2)
                     + 0.6*np.exp(-0.5*((f-1440)/180)**2))
        audio += amplitude*np.sin(2*np.pi*f*t)
    return (audio/max(abs(audio))*0.3).astype(np.float32)


def band_centroid(audio, low, high, rate=48000):
    audio = audio[rate//2:]
    spectrum = np.abs(np.fft.rfft(audio*np.hanning(len(audio))))**2
    frequency = np.fft.rfftfreq(len(audio), 1/rate)
    selected = (frequency >= low) & (frequency <= high)
    return np.sum(frequency[selected]*spectrum[selected])/np.sum(spectrum[selected])


@pytest.mark.parametrize("quality", ["low_latency", "balanced"])
def test_formant_moves_envelope_independently_of_pitch(quality):
    source = vowel()
    base = DSPParameters(mode="female_dsp", pitch=0, formant=0, brightness=50,
                          low_cut=False, quality=quality)
    neutral = render(FemaleDSPProcessor(base), source)
    raised = render(FemaleDSPProcessor(replace(base, formant=4)), source)
    lowered = render(FemaleDSPProcessor(replace(base, formant=-4)), source)
    centers = [band_centroid(a, 350, 1150) for a in (lowered, neutral, raised)]
    assert centers[0] < centers[1]-50 < centers[2]-100
    # Formant-only change retains the 120 Hz harmonic grid; it does not move
    # partial frequencies like a playback-rate change would.
    for audio in (raised, lowered):
        dominant = dominant_frequency(audio)
        assert abs(dominant/120-round(dominant/120)) < 0.02


@pytest.mark.parametrize("quality", ["low_latency", "balanced"])
def test_pitch_compensation_retains_formant_envelope(quality):
    source = vowel()
    base = DSPParameters(mode="female_dsp", pitch=0, formant=0, brightness=50,
                          low_cut=False, quality=quality)
    neutral = render(FemaleDSPProcessor(base), source)
    pitched = render(FemaleDSPProcessor(replace(base, pitch=4)), source)
    original_center = band_centroid(neutral, 350, 1150)
    shifted_center = band_centroid(pitched, 350, 1150)
    # Approximate envelope preservation, allowing resolution/partial spacing.
    assert abs(shifted_center-original_center) < 110


@pytest.mark.parametrize("name", list(PRESETS))
def test_preset_loading_and_output(name):
    parameters = load_preset(name)
    processor = FemaleDSPProcessor(parameters)
    audio = vowel(seconds=0.15)
    result = render(processor, audio)
    assert np.isfinite(result).all() and np.max(np.abs(result)) <= 1
    assert len(result) == len(audio)
    if name != "Original":
        assert not np.array_equal(result, audio)


def test_preset_validation_and_independent_parameters():
    with pytest.raises(ValueError):
        load_preset("Unknown")
    p = FemaleDSPProcessor(load_preset("Female Soft"))
    p.update(pitch=8)
    assert p.parameters.formant == 1.5
    p.update(formant=-3)
    assert p.parameters.pitch == 8
    assert pitch_factor(12) == 2 and formant_factor(0) == 1


def test_live_switching_retains_native_buffers_and_stream():
    dsp = FemaleDSPProcessor()
    backend = FakeBackend()
    engine = AudioEngine(ProcessorChain(main_processor=dsp), backend)
    config = EngineConfig(INPUT, OUTPUT)
    try:
        for _ in range(10):
            engine.start(config)
            native_input, native_output = dsp._native.input, dsp._native.output
            for repeat in range(100):
                dsp.set_parameters(load_preset(list(PRESETS)[repeat % 4]))
                dsp.update(pitch=(repeat % 241)/10-12, formant=(repeat % 121)/10-6,
                           brightness=repeat % 101)
                backend.streams[-1].tick()
                assert np.isfinite(backend.streams[-1].outdata).all()
            assert dsp._native.input is native_input and dsp._native.output is native_output
            assert engine.running
            engine.stop()
        assert len(backend.streams) == 10
    finally:
        engine.stop()


@pytest.mark.parametrize("limiter", [True, False])
def test_limiter_and_nonfinite_safety(limiter):
    processor = FemaleDSPProcessor(DSPParameters(mode="female_dsp", pitch=12, formant=6,
                                   brightness=100, limiter=limiter))
    source = np.tile(np.array([100, -100, np.inf, -np.inf, np.nan, 0], dtype=np.float32), 4000)
    result = render(processor, source)
    assert result.dtype == np.float32 and np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 1
    if limiter:
        assert np.max(np.abs(result[4800:])) <= 0.951


def test_dry_wet_has_algorithmic_delay_alignment():
    rate = 48000
    source = (np.random.default_rng(4).normal(0, 0.1, 8192)).astype(np.float32)
    processor = FemaleDSPProcessor(DSPParameters(mode="female_dsp", wet=0, low_cut=False))
    result = render(processor, source)
    delay = processor.algorithmic_latency_samples
    # After the enable crossfade, wet=0 is delayed original with no filtering.
    np.testing.assert_allclose(result[delay+1024:], source[1024:-delay], atol=1e-7)


def test_female_settings_roundtrip_and_mvp_upgrade(tmp_path):
    manager = SettingsManager(tmp_path / "settings.json")
    settings = AppSettings(dsp_mode="female_dsp", preset="Female Soft", pitch_semitones=3,
                           formant_semitones=1.5, brightness=55, low_cut=False,
                           limiter=True, wet=0.75, dsp_quality="balanced")
    assert manager.save(settings) and manager.load() == settings
    old = AppSettings.from_dict({"gain_db": 4, "buffer_size": 128})
    assert old.dsp_mode == "original" and old.preset == "Original"
    assert old.gain_db == 4 and old.buffer_size == 128
    assert old.dsp_parameters() == DSPParameters()


def test_invalid_dsp_settings_fallback():
    settings = AppSettings.from_dict({"dsp_mode": "AI", "preset": [], "pitch_semitones": 20,
        "formant_semitones": float("nan"), "brightness": "100", "low_cut": "ON",
        "wet": -1, "dsp_quality": {}, "limiter": False})
    assert settings.dsp_parameters() == replace(DSPParameters(), limiter=False)


def test_performance_counters_and_algorithm_delay():
    dsp = FemaleDSPProcessor(load_preset("Female Soft"))
    backend = FakeBackend()
    engine = AudioEngine(ProcessorChain(main_processor=dsp), backend)
    try:
        engine.start(EngineConfig(INPUT, OUTPUT))
        for _ in range(100):
            backend.streams[0].tick()
        main = engine.chain.performance.snapshot()
        callback = engine.performance.snapshot()
        assert main.count == callback.count == 101
        assert 0 < main.average_ms <= main.maximum_ms
        assert callback.average_ms >= main.average_ms
        assert dsp.algorithmic_latency_samples == 2048
    finally:
        engine.stop()


def test_low_cut_reduces_rumble_and_preserves_voice_band():
    rate = 48000
    t = np.arange(rate)/rate
    base = DSPParameters(mode="female_dsp", pitch=0, formant=0, brightness=50,
                         low_cut=False)
    for frequency, minimum, maximum in ((30, 0.0, 0.2), (1000, 0.95, 1.05)):
        audio = (0.1*np.sin(2*np.pi*frequency*t)).astype(np.float32)
        neutral = render(FemaleDSPProcessor(base), audio)[rate//2:]
        filtered = render(FemaleDSPProcessor(replace(base, low_cut=True)), audio)[rate//2:]
        ratio = np.linalg.norm(filtered)/np.linalg.norm(neutral)
        assert minimum <= ratio <= maximum


def test_brightness_controls_high_band_with_limited_boost():
    rate = 48000
    t = np.arange(rate)/rate
    audio = (0.05*np.sin(2*np.pi*7000*t)).astype(np.float32)
    base = DSPParameters(mode="female_dsp", pitch=0, formant=0, brightness=50, low_cut=False)
    levels = [np.linalg.norm(render(FemaleDSPProcessor(replace(base, brightness=b)), audio)[rate//2:])
              for b in (0, 50, 100)]
    assert levels[0] < levels[1] < levels[2]
    assert levels[2]/levels[1] < 1.8


def test_live_parameter_step_remains_finite_and_bounded():
    processor = FemaleDSPProcessor(load_preset("Female Soft"))
    processor.prepare(48000, 256)
    audio = (0.2*np.sin(2*np.pi*220*np.arange(48000)/48000)).astype(np.float32)
    output = []
    for offset in range(0, len(audio), 256):
        if offset % 2048 == 0:
            processor.update(pitch=3 if offset % 4096 else 6, formant=1.5 if offset % 4096 else 4)
        block = audio[offset:offset+256].reshape(-1, 1).copy()
        output.append(processor.process(block, 48000)[:, 0].copy())
    output = np.concatenate(output)
    assert np.isfinite(output).all() and np.max(np.abs(output)) <= 1
    # Catch a full-scale pop/discontinuity on this bounded sine fixture.
    assert np.max(np.abs(np.diff(output))) < 0.12


def test_cpu_load_device_error_stops_engine(monkeypatch):
    def disconnected(stream):
        raise RuntimeError("Audio device disconnected")
    monkeypatch.setattr(FakeStream, "cpu_load", property(disconnected), raising=False)
    engine = AudioEngine(backend=FakeBackend())
    engine.start(EngineConfig(INPUT, OUTPUT))
    with pytest.raises(RuntimeError, match="disconnected"):
        engine.poll()
    assert not engine.running


def test_deadline_exceeded_warning_statistics_reset():
    from src.audio.performance import TimingStats
    stats = TimingStats()
    stats.record(6_000_000, 256, 48000)
    stats.record(1_000_000, 256, 48000)
    assert stats.snapshot().deadline_exceeded == 1
    assert stats.snapshot().average_ms == 3.5 and stats.snapshot().maximum_ms == 6
    stats.reset()
    assert stats.snapshot().count == stats.snapshot().deadline_exceeded == 0
