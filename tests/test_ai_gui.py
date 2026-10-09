import time
from dataclasses import replace

import pytest
from PySide6.QtWidgets import QApplication
from src.audio.controller import AudioController
from src.audio.engine import AudioEngine
from src.processors.chain import ProcessorChain
from src.processors.voice_router import VoiceRouter
from src.processors.ai_voice import AIVoiceProcessor
from src.gui.main_window import MainWindow
from src.settings.manager import SettingsManager
from src.vc.bridge import AIBridge
from src.vc.models import DEFAULT_VOICE_ID, profile
from .fakes import FakeBackend
from .test_ai_voice import FakeClient


def pump(app, predicate):
    until = time.monotonic()+5
    while time.monotonic()<until:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('GUI did not finish operation')


@pytest.fixture
def ai_window(tmp_path):
    app = QApplication.instance() or QApplication([])
    bridge = AIBridge(client_factory=FakeClient)
    router = VoiceRouter(ai=AIVoiceProcessor(bridge=bridge))
    backend = FakeBackend()
    controller = AudioController(AudioEngine(ProcessorChain(router), backend))
    window = MainWindow(SettingsManager(tmp_path/'settings.json'), controller)
    window.show()
    pump(app, lambda: window.start_button.isEnabled())
    yield app, window, backend, bridge
    window.close()
    pump(app, lambda: not controller.alive)
    app.processEvents()
    assert not bridge.alive


def test_ai_gui_lazy_loading_mode_controls_settings_and_stop(ai_window):
    app, window, backend, bridge = ai_window
    # The app starts on the shipped voice, so the worker is loaded rather than waiting
    # for the user to pick a mode first.
    pump(app, lambda: bridge.status == 'Ready')
    assert [window.mode.itemData(i) for i in range(3)] == ['original','female_dsp','ai_voice']
    assert bridge.parameters.model == DEFAULT_VOICE_ID
    # The DSP controls belong to the Female DSP path, which is not the active mode.
    assert not window.pitch.isEnabled()
    # Neither the DSP nor the AI brightness control applies to the shipped route.
    assert not window.brightness.isEnabled() and not window.ai_brightness.isEnabled()
    window.start_button.click()
    pump(app, lambda: window.stop_button.isEnabled())
    assert not window.ai_quality.isEnabled()
    # The shipped route has no brightness stage, so the control is disabled and the
    # worker keeps its neutral value rather than accepting an ignored request.
    assert not window.ai_brightness.isEnabled()
    assert bridge.parameters.brightness == 50
    for _ in range(10):
        backend.streams[-1].tick()
        app.processEvents()
    window.stop_button.click()
    pump(app, lambda: window.start_button.isEnabled())
    assert not bridge.alive
    window.close()
    pump(app, lambda: not window.controller.alive)
    saved = window.manager.load()
    # The shipped route has no brightness stage, so the neutral value is what persists.
    assert saved.voice_mode == 'ai_voice' and saved.ai_brightness == 50


def test_ai_gui_quality_changes_are_acknowledged_off_gui_thread(ai_window):
    app, window, backend, bridge = ai_window
    window.mode.setCurrentIndex(2)
    pump(app, lambda: bridge.status == 'Ready')
    window.ai_quality.setCurrentIndex(window.ai_quality.findData('stable'))
    assert window._pending
    pump(app, lambda: not window._pending and bridge.status == 'Ready')
    # Quality is acknowledged off the GUI thread even though the route ignores it,
    # and the ring capacity stays fixed at the shipped route's sizing.
    assert bridge.parameters.quality == 'stable'
    assert bridge.input.capacity >= bridge.chunk_frames*profile(DEFAULT_VOICE_ID)['queue_chunks']
    assert not window.ai_brightness.isEnabled()
    window.mode.setCurrentIndex(0)
    window.mode.setCurrentIndex(window.mode.findData('female_dsp'))
    assert window.pitch.isEnabled()


def test_utterance_lavasr_is_selectable_saved_and_locked_while_running(ai_window):
    app,window,backend,bridge=ai_window
    window.ai_delivery.setCurrentIndex(window.ai_delivery.findData('utterance_lavasr'))
    pump(app,lambda:not window._pending)
    assert bridge.parameters.delivery=='utterance' and bridge.parameters.enhancer=='lavasr'
    assert bridge.parameters.lavasr_denoise is True
    window.mode.setCurrentIndex(2)
    pump(app,lambda:bridge.status=='Ready')
    window.start_button.click();pump(app,lambda:window.stop_button.isEnabled())
    assert not window.ai_delivery.isEnabled() and window.finish_phrase.isEnabled()
    window.stop_button.click();pump(app,lambda:window.start_button.isEnabled())
    window._capture_settings()
    assert window.settings.ai_enhancer=='lavasr' and window.settings.ai_delivery=='utterance'
    assert window.settings.ai_lavasr_denoise is True


def test_recommended_preserves_devices_and_snapshot_is_async(ai_window):
    import json
    app,window,backend,bridge=ai_window
    input_device,output_device=window.input_device.currentData(),window.output_device.currentData()
    window.ai_wait.setValue(25)
    pump(app,lambda:not window._pending)
    window.ai_fade.setValue(50)
    pump(app,lambda:not window._pending)
    window.ai_pitch.setValue(24)
    window.recommended.click()
    pump(app,lambda:not window._pending)
    assert window.input_device.currentData()==input_device
    assert window.output_device.currentData()==output_device
    assert window.rate.currentData()==48000 and window.buffer.currentData()==256
    # The shipped route runs natively at pitch 0 and neutral brightness.
    assert bridge.parameters.pitch==0 and bridge.parameters.brightness==50
    assert bridge.parameters.crossfade_ms==20
    window.debug_snapshot.click()
    folder=window.manager.path.parent/'debug_snapshots'
    pump(app,lambda:bool(list(folder.glob('*.json'))) if folder.exists() else False)
    data=json.loads(next(folder.glob('*.json')).read_text(encoding='utf-8'))
    assert data['settings']['ai_model']==DEFAULT_VOICE_ID
    assert 'p99_ms' in data['callback'] and 'worker_state' in data['ai']


def test_ai_restart_remains_nonblocking_and_retains_engine(ai_window):
    app,window,backend,bridge=ai_window
    window.mode.setCurrentIndex(2)
    pump(app,lambda:bridge.status=='Ready')
    window.start_button.click();pump(app,lambda:window.stop_button.isEnabled())
    old=bridge.thread
    window.ai_restart.click()
    pump(app,lambda:not window._pending and bridge.status=='Ready' and bridge.thread is not old)
    assert window.controller.engine.running and not old.is_alive()
    assert not window.ai_wait.isEnabled() and not window.ai_pitch.isEnabled()


def test_mme_warning_and_wasapi_first_without_changing_selection(ai_window):
    from dataclasses import replace
    app,window,backend,bridge=ai_window
    originals=window.controller.devices
    mme=tuple(replace(d,index=d.index+10,host_api='MME') for d in originals)
    window.controller.devices=mme+originals
    window.settings.input_device=mme[0].identity
    window.settings.output_device=mme[-1].identity
    window._populate_devices()
    assert window.input_device.itemData(0).host_api=='Windows WASAPI'
    assert window.input_device.currentData().host_api=='MME'
    assert 'Windows WASAPIにしてください' in window.route_hint.text()


def test_device_disconnect_stops_worker_and_recovers_without_restarting_gui(ai_window):
    app,window,backend,bridge=ai_window
    window.mode.setCurrentIndex(2);pump(app,lambda:bridge.status=='Ready')
    window.start_button.click();pump(app,lambda:window.stop_button.isEnabled())
    backend.streams[-1].active=False  # simulated driver disappearance
    pump(app,lambda:window.controller.snapshot.state=='Error' and not bridge.alive)
    assert 'disconnected' in window.controller.snapshot.error
    assert window.controller.alive and not window.controller.engine.running
    window.refresh.click();pump(app,lambda:window.start_button.isEnabled())
    window.start_button.click();pump(app,lambda:window.stop_button.isEnabled() and bridge.status=='Ready')
    assert window.controller.engine.running and bridge.alive


def test_recommended_settings_preserve_devices_and_block_live_model_change(ai_window):
    app,window,backend,bridge=ai_window
    selected=(window.input_device.currentIndex(),window.output_device.currentIndex())
    window.recommended.click()
    pump(app,lambda:not window._pending)
    value=window._ai_parameters()
    assert value.model==DEFAULT_VOICE_ID
    assert window.rate.currentData()==48000 and window.buffer.currentData()==256
    assert selected==(window.input_device.currentIndex(),window.output_device.currentIndex())
    window.start_button.click()
    pump(app,lambda:window.stop_button.isEnabled())
    assert not window.ai_model.isEnabled()
    assert not window.ai_pitch.isEnabled()


def _wait_guide(app, window, show, timeout=2.0):
    """Pump events until the guide opens (or closes), or the bound expires."""
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        app.processEvents()
        if getattr(window, '_guide_open', False) is show:
            return True
        time.sleep(.005)
    return False


def test_first_run_guide_waits_for_the_splash_instead_of_covering_it(ai_window, monkeypatch):
    """The guide is its own window, so the splash overlay cannot hide it.

    Scheduled before the animation ended, it appeared on top of the splash: two
    startup messages at once. It now waits for the splash to reveal the window, and
    a host that plays no splash opens it right away.
    """
    from src.gui import splash

    app, window, _backend, _bridge = ai_window
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    window.settings = replace(window.settings, tutorial_seen=False)
    monkeypatch.setattr(splash, 'splash_duration_ms', lambda: 250)
    window._schedule_first_run_tutorial()
    # Still inside the splash window: nothing on top of the animation yet.
    app.processEvents()
    assert window._guide_open is False
    assert _wait_guide(app, window, True), 'the guide never followed the splash'
    window._guide_dialog.close()
    assert _wait_guide(app, window, False)

    # Without a splash to play there is nothing to wait for.
    monkeypatch.setattr(splash, 'splash_duration_ms', lambda: 0)
    window.settings = replace(window.settings, tutorial_seen=False)
    window._schedule_first_run_tutorial()
    assert _wait_guide(app, window, True)
    window._guide_dialog.close()
    assert _wait_guide(app, window, False)
