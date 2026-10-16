from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.settings.manager import AppSettings, SettingsManager


def test_settings_roundtrip(tmp_path):
    manager = SettingsManager(tmp_path / "nested" / "settings.json")
    settings = AppSettings(input_device={"name": "USB マイク", "host_api": "Windows WASAPI"},
                           output_device={"name": "CABLE Input", "host_api": "Windows WASAPI"},
                           gain_db=4.5, noise_gate_db=-50, sample_rate=44100, buffer_size=512,
                           window_size=(600, 800), window_position=(-100, 40))
    assert manager.save(settings)
    assert manager.load() == settings
    assert not (tmp_path / "nested" / "settings.json.tmp").exists()


def test_finish_formant_and_level_settings_fall_back():
    from src.settings.manager import AppSettings
    settings = AppSettings.from_dict({"finish": "yes", "ai_formant_on": 1,
                                      "ai_formant": 99, "auto_level": "yes"})
    assert settings.finish is False
    assert settings.ai_formant_on is False and settings.ai_formant == 2.0
    assert settings.auto_level is False
    assert AppSettings().finish is False
    assert AppSettings().ai_formant_on is False
    assert AppSettings().auto_level is False


def test_env_passthrough_defaults_on_and_rejects_non_booleans():
    assert AppSettings().env_passthrough is True
    assert AppSettings.from_dict({"env_passthrough": False}).env_passthrough is False
    assert AppSettings.from_dict({"env_passthrough": "yes"}).env_passthrough is True


def test_the_initial_ai_thread_count_is_two():
    """`_apply_recommended` picks 2 threads for the shipped MeanVC2 voice, so a fresh
    install must not start one step behind it. Missing or invalid values fall back to
    the same default, not a different one.
    """
    assert AppSettings().ai_threads == 2
    assert AppSettings.from_dict({"ai_threads": True}).ai_threads == 2
    assert AppSettings.from_dict({"ai_threads": 9}).ai_threads == 2
    # A stored choice is still honoured.
    assert AppSettings.from_dict({"ai_threads": 4}).ai_threads == 4


@pytest.mark.parametrize("text", ["", "broken", "[]", "null", "42", "{", '{"gain_db": NaN}',
                                   '{"buffer_size": true}', '{"sample_rate": "48000"}'])
def test_corrupt_or_invalid_settings(tmp_path, text):
    path = tmp_path / "settings.json"
    path.write_text(text, encoding="utf-8")
    assert SettingsManager(path).load() == AppSettings()


def test_missing_settings(tmp_path):
    assert SettingsManager(tmp_path / "missing.json").load() == AppSettings()


@pytest.mark.parametrize("value", [None, True, "loud", float("inf"), float("nan"), 1e200, 10**400, [], {}])
def test_invalid_numeric_fields(value):
    settings = AppSettings.from_dict({"gain_db": value, "noise_gate_db": value})
    assert settings.gain_db == 0 and settings.noise_gate_db == -45


def test_invalid_device_geometry_monitor():
    settings = AppSettings.from_dict({"input_device": {"name": 4}, "output_device": 9,
                                     "window_size": [-1, 4], "window_position": "oops",
                                     "monitor": True, "monitor_volume_db": "loud",
                                     "unknown": "ignored"})
    # Invalid geometry falls back to the defaults, while the monitor switch is a
    # user setting the app now honours (with its level falling back too).
    assert settings == replace(AppSettings(), monitor=True)
    assert settings.monitor_volume_db == 0.0


def test_partial_invalid_settings_preserves_valid_fields():
    settings = AppSettings.from_dict({"gain_db": -6, "buffer_size": 900, "noise_gate_db": -20,
                                     "sample_rate": 44100})
    assert settings.gain_db == -6 and settings.noise_gate_db == -20
    assert settings.sample_rate == 44100 and settings.buffer_size == 256


def test_save_failure_preserves_old_file(tmp_path, monkeypatch):
    manager = SettingsManager(tmp_path / "settings.json")
    assert manager.save(AppSettings(gain_db=2))
    def fail(*args, **kwargs):
        raise PermissionError("Read-only")
    monkeypatch.setattr(Path, "replace", fail)
    assert not manager.save(AppSettings(gain_db=7))
    assert manager.load().gain_db == 2


def test_large_settings_rejected(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(" " * 65537)
    assert SettingsManager(path).load() == AppSettings()
