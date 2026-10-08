"""Run these dependency-specific tests in .venv-ai; main app stays AI-free."""
import numpy as np
import pytest

pytest.importorskip('scipy', reason='Numeric worker tests require the isolated AI environment')
from src.vc.resampler import StreamingResampler
from src.vc.post_fx import LightPostFX


@pytest.mark.parametrize('rates', [(48000, 16000), (16000, 48000)])
def test_resampler_continuity_with_arbitrary_chunk_boundaries(rates):
    audio = np.random.default_rng(24).normal(0, .1, 4801).astype(np.float32)
    whole = StreamingResampler(*rates).process(audio)
    streamed = StreamingResampler(*rates)
    pieces = []
    offset = 0
    for length in [1, 2, 64, 128, 256, 512, 1024, 2814]:
        pieces.append(streamed.process(audio[offset:offset+length]))
        offset += length
    pieces.append(streamed.process(audio[offset:]))
    result = np.concatenate(pieces)
    assert result.dtype == np.float32 and np.isfinite(result).all()
    assert len(result) == ((len(audio)+2)//3 if rates[0] == 48000 else 3*len(audio))
    np.testing.assert_allclose(result, whole, rtol=1e-5, atol=1e-7)
    streamed.reset()
    np.testing.assert_array_equal(streamed.process(audio), whole)


def test_resampler_antialiasing_and_passband():
    t = np.arange(48000, dtype=np.float32)/48000
    low = StreamingResampler(48000,16000).process(np.sin(2*np.pi*1000*t).astype(np.float32))
    high = StreamingResampler(48000,16000).process(np.sin(2*np.pi*12000*t).astype(np.float32))
    assert np.sqrt(np.mean(low[500:]**2)) > .69
    assert np.sqrt(np.mean(high[500:]**2)) < .001


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_resampler_rejects_nonfinite(bad):
    with pytest.raises(ValueError):
        StreamingResampler(48000,16000).process(np.array([bad], dtype=np.float32))


@pytest.mark.parametrize('brightness', [0, 50, 100])
def test_post_fx_safety_and_reset(brightness):
    fx = LightPostFX()
    audio = np.array([np.nan, np.inf, -np.inf, 100, -100, .1]*128, dtype=np.float32)
    output = fx.process(audio, brightness, True, True)
    assert len(output) == len(audio) and output.dtype == np.float32
    assert np.isfinite(output).all() and np.max(np.abs(output)) <= .950001
    fx.reset()
    np.testing.assert_array_equal(output, fx.process(audio, brightness, True, True))


def test_low_cut_reduces_rumble_without_removing_voice():
    t = np.arange(48000, dtype=np.float32)/48000
    fx = LightPostFX()
    low = fx.process((.1*np.sin(2*np.pi*20*t)).astype(np.float32), low_cut=True)
    fx.reset()
    voice = fx.process((.1*np.sin(2*np.pi*1000*t)).astype(np.float32), low_cut=True)
    assert np.sqrt(np.mean(low[24000:]**2)) < .004
    assert np.sqrt(np.mean(voice[24000:]**2)) > .069


def test_post_fx_bypass_and_malicious_amplitude_stay_safe():
    source = np.array([.1, -.2, .3], dtype=np.float32)
    np.testing.assert_array_equal(LightPostFX().process(source, enabled=False, limiter=False), source)
    output = LightPostFX().process(np.array([np.finfo(np.float32).max], dtype=np.float32))
    assert np.isfinite(output).all() and abs(output[0]) <= .95


@pytest.mark.parametrize('gain',[-100,0,100,np.nan,np.inf])
def test_prosody_gain_before_limiter_safe(gain):
    fx=LightPostFX()
    source=np.full(624,2,np.float32)
    output=fx.process(source,enabled=False,dynamic_gain_db=gain)
    assert output.dtype==np.float32 and np.isfinite(output).all() and np.max(np.abs(output))<=.950001
    assert abs(fx.dynamic_gain_db)<=1.5


def test_prosody_off_keeps_existing_fx_identical():
    source=np.random.default_rng(5).normal(0,.1,624).astype(np.float32)
    np.testing.assert_array_equal(LightPostFX().process(source),LightPostFX().process(source,dynamic_gain_db=0))
