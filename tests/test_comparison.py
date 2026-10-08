import numpy as np
import pytest

from tools.compare_presets import read_wav, render, write_wav
from src.presets.female_presets import load_preset


@pytest.mark.parametrize("preset", ["Original", "Female Soft", "Female Bright", "Anime Test"])
def test_comparison_files_keep_duration_and_valid_audio(tmp_path, preset):
    signal = (0.1*np.sin(2*np.pi*220*np.arange(16000)/48000)).astype(np.float32)
    output, delay, timing = render(signal, 48000, load_preset(preset))
    assert output.dtype == np.float32 and len(output) == len(signal)
    assert np.isfinite(output).all() and np.max(np.abs(output)) <= 1
    assert timing.count > 0 and delay == (0 if preset == "Original" else 2048)
    path = tmp_path / "comparison.wav"
    write_wav(path, output, 48000)
    recovered, rate = read_wav(path)
    assert rate == 48000 and len(recovered) == len(signal)
    np.testing.assert_allclose(recovered, output, atol=1/16000)
