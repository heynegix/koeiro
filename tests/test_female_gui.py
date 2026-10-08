import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.audio.controller import AudioController
from src.audio.engine import AudioEngine
from src.processors.chain import ProcessorChain
from src.processors.female_dsp import FemaleDSPProcessor
from src.gui.main_window import MainWindow
from src.settings.manager import SettingsManager
from .fakes import FakeBackend
from .test_gui import pump


@pytest.fixture
def dsp_window(tmp_path):
    app = QApplication.instance() or QApplication([])
    backend = FakeBackend()
    dsp = FemaleDSPProcessor()
    controller = AudioController(AudioEngine(ProcessorChain(main_processor=dsp), backend))
    window = MainWindow(SettingsManager(tmp_path / "settings.json"), controller)
    window.show()
    pump(app, lambda: window.start_button.isEnabled())
    yield app, window, backend
    window.close()
    pump(app, lambda: not controller.alive)


def test_female_gui_live_controls_presets_bypass_and_persistence(dsp_window):
    app, window, backend = dsp_window
    # The app starts on the shipped voice; this page is reached by selecting the DSP mode.
    window.mode.setCurrentIndex(window.mode.findData("female_dsp"))
    assert window.dsp.parameters.mode == "female_dsp"
    window.preset.setCurrentText("Female Soft")
    assert window.dsp.parameters.pitch == 3 and window.dsp.parameters.formant == 1.5
    window.start_button.click()
    pump(app, lambda: window.stop_button.isEnabled())
    assert not window.quality.isEnabled()
    for i in range(100):
        window.preset.setCurrentText(("Original", "Female Bright", "Anime Test")[i % 3])
        window.pitch.setValue(i % 241-120)
        window.formant.setValue(i % 121-60)
        window.brightness.setValue(i % 101)
        backend.streams[0].tick()
        assert np.isfinite(backend.streams[0].outdata).all()
    assert len(backend.streams) == 1
# The app now starts on the shipped voice, so the DSP is reached explicitly rather
    # than by toggling the bypass button away from it.
    window.mode.setCurrentIndex(window.mode.findData("female_dsp"))
    assert window.mode.currentData() == "female_dsp"
    window.stop_button.click()
    pump(app, lambda: window.start_button.isEnabled())
    window.quality.setCurrentIndex(1)
    window.pitch.setValue(32)
    window.formant.setValue(17)
    window.low_cut.setChecked(False)
    window.wet.setValue(80)
    window.close()
    pump(app, lambda: not window.controller.alive)
    settings = window.manager.load()
    assert settings.pitch_semitones == 3.2 and settings.formant_semitones == 1.7
    assert settings.dsp_quality == "balanced" and not settings.low_cut and settings.wet == 0.8
