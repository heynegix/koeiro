import pytest
from PySide6.QtWidgets import QApplication, QDialog

from src.gui.tutorial import PAGES, TutorialDialog
from tests.test_ai_gui import ai_window, pump  # noqa: F401  (ai_window is used as a fixture)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_tutorial_has_six_pages(qapp):
    dialog = TutorialDialog()
    try:
        assert dialog.stack.count() == 6
        assert len(PAGES) == 6
        assert dialog.page == 0
    finally:
        dialog.close()


def test_tutorial_navigation_and_finish(qapp):
    dialog = TutorialDialog()
    try:
        assert not dialog.back_button.isEnabled()
        for _ in range(5):
            dialog.next_button.click()
        assert dialog.page == 5
        assert dialog.next_button.text() == "はじめる"
        assert dialog.back_button.isEnabled()
        dialog.next_button.click()
        assert dialog.result() == QDialog.DialogCode.Accepted
    finally:
        dialog.close()


def test_tutorial_back_and_skip(qapp):
    dialog = TutorialDialog()
    try:
        dialog.next_button.click()
        dialog.next_button.click()
        assert dialog.page == 2
        dialog.back_button.click()
        assert dialog.page == 1
        assert not dialog.hide_next_time
        dialog.hide_box.setChecked(True)
        assert dialog.hide_next_time
        dialog.skip_button.click()
        assert dialog.result() == QDialog.DialogCode.Rejected
    finally:
        dialog.close()


def test_tutorial_seen_roundtrip():
    from src.settings.manager import AppSettings
    assert AppSettings().tutorial_seen is False
    assert AppSettings.from_dict({"tutorial_seen": True}).tutorial_seen is True
    assert AppSettings.from_dict({"tutorial_seen": "yes"}).tutorial_seen is False


def test_monitor_settings_roundtrip():
    from src.settings.manager import AppSettings
    assert AppSettings().monitor is False
    assert AppSettings().monitor_device is None
    assert AppSettings().monitor_volume_db == 0.0
    values = AppSettings.from_dict({"monitor": True, "monitor_volume_db": -9.5,
                                    "monitor_device": {"name": "HP", "host_api": "X"}})
    assert values.monitor is True
    assert values.monitor_device == {"name": "HP", "host_api": "X"}
    assert values.monitor_volume_db == -9.5
    assert AppSettings.from_dict({"monitor": "yes"}).monitor is False
    assert AppSettings.from_dict({"monitor_volume_db": 99}).monitor_volume_db == 0.0
    assert AppSettings.from_dict({"monitor_volume_db": "loud"}).monitor_volume_db == 0.0
    params = values.ai_parameters()
    assert params is not None


def test_first_run_tutorial_marks_seen(ai_window, monkeypatch):
    """A reader who finishes (or hides) the guide does not see it again."""
    from src.gui import tutorial as tutorial_module
    app, window, backend, bridge = ai_window
    calls = []
    real = tutorial_module.TutorialDialog

    class StubDialog(real):
        def __init__(self, parent=None, on_page=None):
            calls.append(parent)
            super().__init__(parent, on_page=on_page)
            self.hide_box.setChecked(True)

        def show(self):
            self.accept()

    monkeypatch.setattr(tutorial_module, "TutorialDialog", StubDialog)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert window.settings.tutorial_seen is False
    window._maybe_show_tutorial()
    assert calls
    assert window.settings.tutorial_seen is True
    assert window._guide_open is False


def test_no_tutorial_when_already_seen(ai_window, monkeypatch):
    from dataclasses import replace
    from src.gui import tutorial as tutorial_module
    app, window, backend, bridge = ai_window
    window.settings = replace(window.settings, tutorial_seen=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("tutorial must not open")

    monkeypatch.setattr(tutorial_module, "TutorialDialog", forbidden)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    window._maybe_show_tutorial()


def test_first_run_guide_skip_keeps_it_for_the_next_launch(ai_window, monkeypatch):
    """Skipping closes the guide without remembering it; hiding remembers it."""
    from src.gui import tutorial as tutorial_module
    app, window, backend, bridge = ai_window
    real = tutorial_module.TutorialDialog

    class StubDialog(real):
        def show(self):
            self.reject()

    monkeypatch.setattr(tutorial_module, "TutorialDialog", StubDialog)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    window._maybe_show_tutorial()
    assert window.settings.tutorial_seen is False
    assert window._guide_open is False


def test_guide_button_is_pinned_to_the_bottom_right(ai_window):
    app, window, backend, bridge = ai_window
    window.resize(1200, 820)
    window.show()
    app.processEvents()
    window._place_help_button()
    container = window.centralWidget()
    assert window.help_button.isEnabled()
    assert window.help_button.x() + window.help_button.width() <= container.width()
    assert window.help_button.y() + window.help_button.height() <= container.height()
    assert window.help_button.x() > container.width()*0.5
    assert window.help_button.y() > container.height()*0.5


def test_guide_walks_the_window_and_highlights_each_control(ai_window):
    app, window, backend, bridge = ai_window
    dialog = TutorialDialog(window, on_page=window._guide_page)
    try:
        assert window._active_page == 'home'
        dialog.next_button.click()
        assert window._active_page == 'home'
        assert window.input_device.property('guide') is True
        # ② mode/voice and ③ Start both stay on the home page, so the guide only
        # changes page when it explains the delivery route.
        dialog.next_button.click()
        dialog.next_button.click()
        assert window._active_page == 'home'
        assert window.start_button.property('guide') is True
        dialog.next_button.click()
        assert window._active_page == 'library'
        assert window.ai_delivery.property('guide') is True
        assert window.input_device.property('guide') is False
        dialog.skip_button.click()
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert window.settings.tutorial_seen is False
    finally:
        dialog.close()


def test_closing_the_guide_clears_the_highlight_and_keeps_it_for_next_launch(ai_window, monkeypatch):
    from src.gui import tutorial as tutorial_module
    app, window, backend, bridge = ai_window
    real = tutorial_module.TutorialDialog

    class Scripted(real):
        """A reader who turns three pages and then closes the guide."""

        def show(self):
            super().show()
            for _ in range(4):
                self.next_button.click()
            self.reject()

    monkeypatch.setattr(tutorial_module, 'TutorialDialog', Scripted)
    window._show_tutorial()
    assert window._active_page == 'library'
    assert window.ai_delivery.property('guide') is False
    assert window.settings.tutorial_seen is False


def test_default_voice_shows_as_the_app_standard(ai_window):
    from src.vc.voice_library import DEFAULT_VOICE_ID, DEFAULT_VOICE_NAME
    app, window, backend, bridge = ai_window
    assert window.ai_model.itemData(0) == DEFAULT_VOICE_ID
    window.ai_model.setCurrentIndex(0)
    window._update_voice_info()
    assert DEFAULT_VOICE_NAME in window.voice_hint.text()
    assert "アプリ標準の声" in window.voice_hint.text()
    assert "登録済み" not in window.voice_hint.text()
    # Thread count still applies to the standard voice; the fixed processing chunk does not.
    assert window.ai_threads.isEnabled()
    assert not window.ai_quality.isEnabled()
