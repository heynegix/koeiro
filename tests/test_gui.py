import time
from threading import Event

import pytest
from PySide6.QtWidgets import QApplication

from src.audio.controller import AudioController
from src.audio.engine import AudioEngine
from src.gui.main_window import MainWindow
from src.settings.manager import AppSettings, SettingsManager
from .fakes import FakeBackend, INPUT, OUTPUT


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(qapp, predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("GUI timed out")


@pytest.fixture
def window(qapp, tmp_path):
    backend = FakeBackend()
    controller = AudioController(AudioEngine(backend=backend))
    window = MainWindow(SettingsManager(tmp_path / "settings.json"), controller)
    window.show()
    pump(qapp, lambda: window.start_button.isEnabled())
    yield window, backend
    window.close()
    pump(qapp, lambda: not controller.alive)
    qapp.processEvents()


def test_gui_devices_controls_live_parameters_and_ten_cycles(window, qapp):
    window, backend = window
    assert window.input_device.count() == 1
    assert window.output_device.count() == 1
    assert window.monitor_device.count() == 1
    assert window.monitor_device.currentData() == OUTPUT
    assert window.monitor_toggle.isEnabled()
    assert window.monitor_toggle.text() == "Monitor OFF"
    assert not window.monitor_toggle.isChecked()
    assert window.input_device.currentData() == INPUT
    assert window.output_device.currentData() == OUTPUT
    assert window.input_device.minimumHeight() >= 30
    assert window.rate.minimumHeight() >= 30
    for _ in range(10):
        window.start_button.click()
        window.start_button.click()
        pump(qapp, lambda: window.stop_button.isEnabled())
        assert not window.input_device.isEnabled() and window.gain.isEnabled()
        assert window.controller.engine.running
        backend.streams[-1].tick()
        window.gain.setValue(60)
        window.gate.setValue(-500)
        assert window.controller.engine.chain.gain._target == pytest.approx(10 ** (6 / 20))
        assert window.controller.engine.chain.gate._thresholds[0] == pytest.approx(10 ** (-50 / 20))
        window.stop_button.click()
        pump(qapp, lambda: window.start_button.isEnabled())
        assert not window.controller.engine.running
    assert len(backend.streams) == 10
    assert "実測値ではありません" in window.latency.text()


def test_refresh_waits_for_worker_and_does_not_block_ui(window, qapp, monkeypatch):
    window, backend = window
    released = Event()
    entered = Event()
    original = backend.query_devices
    def delayed(index=None):
        entered.set()
        assert released.wait(2)
        return original(index)
    monkeypatch.setattr(backend, "query_devices", delayed)
    try:
        window.refresh.click()
        pump(qapp, entered.is_set)
        # Pending request must not be acknowledged by the old Stopped snapshot.
        for _ in range(5):
            window._tick()
        assert not window.start_button.isEnabled()
        assert not window.refresh.isEnabled()
        window.gain.setValue(20)
        assert window.gain_label.text() == "+2.0 dB"
    finally:
        released.set()
    pump(qapp, lambda: window.start_button.isEnabled())


def test_error_display_then_restart(window, qapp):
    window, backend = window
    backend.fail_start = True
    window.start_button.click()
    pump(qapp, lambda: "Error" in window.status.text() and window.start_button.isEnabled())
    assert not window.controller.engine.running
    backend.fail_start = False
    window.start_button.click()
    pump(qapp, lambda: window.stop_button.isEnabled())


def test_close_during_running_releases_stream_and_saves(window, qapp):
    window, backend = window
    window.start_button.click()
    pump(qapp, lambda: window.stop_button.isEnabled())
    window.gain.setValue(-40)
    window.close()
    pump(qapp, lambda: not window.controller.alive)
    assert backend.streams[0].closed
    assert window.manager.load().gain_db == -4
    assert window.manager.load().input_device == INPUT.identity


def test_missing_device_and_corrupt_settings_do_not_prevent_gui(qapp, tmp_path):
    class EmptyBackend(FakeBackend):
        def query_devices(self, index=None):
            return []
    path = tmp_path / "settings.json"
    path.write_text("{broken", encoding="utf-8")
    controller = AudioController(AudioEngine(backend=EmptyBackend()))
    window = MainWindow(SettingsManager(path), controller)
    window.show()
    try:
        pump(qapp, lambda: controller.devices_revision == 1 and window._revision == 1)
        assert not window.start_button.isEnabled()
        assert window.input_device.count() == window.output_device.count() == 0
        assert "ありません" in window.route_hint.text()
    finally:
        window.close()
        pump(qapp, lambda: not controller.alive)


def test_monitor_toggle_persists_and_plays_alongside_main_stream(window, qapp):
    window, backend = window
    assert window.monitor_toggle.text() == "Monitor OFF"
    window.monitor_toggle.click()
    assert window.monitor_toggle.text() == "Monitor ON"
    window._capture_settings()
    assert window.settings.monitor is True
    assert window.settings.monitor_device == OUTPUT.identity
    window.start_button.click()
    pump(qapp, lambda: window.stop_button.isEnabled())
    engine = window.controller.engine
    assert engine.monitor_player is not None and engine.monitor_player.running
    pump(qapp, lambda: "Monitor再生中" in window.monitor_note.text())
    backend.streams[0].tick()
    backend.streams[-1].tick()
    assert engine.monitor_player.played > 0
    window.stop_button.click()
    pump(qapp, lambda: window.start_button.isEnabled())
    assert engine.monitor_player is None
    assert window.monitor_toggle.isEnabled()


def test_monitor_switch_and_level_apply_live_without_restarting(window, qapp):
    """The monitor is not a Stop/Start setting: both edits land while running."""
    window, backend = window
    window.monitor_toggle.click()
    window.start_button.click()
    pump(qapp, lambda: window.stop_button.isEnabled())
    engine = window.controller.engine
    assert engine.monitor_player is not None and engine.monitor_player.running
    # The switch stays usable while the engine runs.
    assert window.monitor_toggle.isEnabled() and window.monitor_device.isEnabled()
    window.monitor_toggle.click()
    pump(qapp, lambda: engine.monitor_player is None)
    assert window.monitor_toggle.text() == "Monitor OFF"
    assert engine.running
    window.monitor_toggle.click()
    pump(qapp, lambda: engine.monitor_player is not None and engine.monitor_player.running)
    window.monitor_volume.setValue(-120)
    pump(qapp, lambda: engine.monitor_gain_db == pytest.approx(-12.0))
    assert engine.monitor_player.gain == pytest.approx(10 ** (-12 / 20), rel=1e-6)
    window.monitor_volume.setValue(0)
    pump(qapp, lambda: engine.monitor_gain_db == pytest.approx(0.0))
    assert engine.monitor_player.gain == pytest.approx(1.0)
    window._capture_settings()
    assert window.settings.monitor_volume_db == 0.0
    window.stop_button.click()
    pump(qapp, lambda: window.start_button.isEnabled())


def test_monitor_warns_when_it_shares_the_main_output(window, qapp):
    window, backend = window
    # The fixture has a single output device, so the monitor device is the main output.
    assert window.monitor_device.currentData() == OUTPUT
    assert 'メインの出力先と同じ' in window.monitor_warning.text()
    window.monitor_volume.setValue(-30)
    window.monitor_volume.setValue(0)
    assert window.monitor_volume_label.text() == '+0.0 dB'


def test_monitor_test_tone_reports_through_the_note(window, qapp):
    window, backend = window
    window.monitor_test.click()
    assert 'テスト音' in window.monitor_note.text()
    assert not window.monitor_test.isEnabled()
    pump(qapp, lambda: '再生しました' in window.monitor_note.text())
    assert window.monitor_test.isEnabled()
