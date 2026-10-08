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
