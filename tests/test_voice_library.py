import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.vc.voice_library import discover, folder_for
from tools.register_voice import centroid, select_segments
from tests.test_ai_gui import ai_window, pump


def voice_files(root, identifier='user_'+'a'*32):
    folder = folder_for(identifier, root)
    folder.mkdir(parents=True)
    for name in ('runtime.json', 'reference.wav', 'fixed_embedding.npy'):
        (folder/name).write_bytes(b'fixture')
    info = dict(schema=1, id=identifier, name='試験の声', reference_seconds=5., source_filename='声.m4a')
    (folder/'profile.json').write_text(json.dumps(info), 'utf-8')
    return folder


def test_incomplete_and_unsafe_library_entries_are_not_live_profiles(tmp_path):
    folder = voice_files(tmp_path)
    identifier = folder.name
    profiles = discover(tmp_path)
    assert profiles[identifier]['backend'] == 'meanvc2'
    assert profiles[identifier]['phrase_repair']
    (folder/'fixed_embedding.npy').unlink()
    assert discover(tmp_path) == {}
    with pytest.raises(ValueError):
        folder_for('../../meanvc2_ref20', tmp_path)


def test_a_profile_nested_past_the_parser_limit_is_skipped(tmp_path):
    """discover() runs while src.vc.models is imported. A voice folder holding a
    profile.json of nothing but "[" repeated 30000 times is under the 64 KB size cap
    but past the interpreter's recursion limit, and the RecursionError used to escape
    the import -- the whole app refused to start instead of losing one voice.
    """
    good = voice_files(tmp_path, identifier='user_' + 'b' * 32)
    broken = voice_files(tmp_path, identifier='user_' + 'c' * 32)
    (broken / 'profile.json').write_text('[' * 30000 + ']' * 30000, 'utf-8')
    assert (broken / 'profile.json').stat().st_size < 65536  # the size guard lets it in
    assert set(discover(tmp_path)) == {good.name}


def test_the_voice_panel_says_what_it_could_not_read(ai_window, monkeypatch):
    """The panel quoted profile.json straight from the slot. An unreadable file raised,
    Qt swallowed it, and the screen kept the previous voice's numbers.
    """
    app, window, backend, bridge = ai_window

    def unreadable(*args, **kwargs):
        raise RecursionError('profile.json nested past the parser limit')

    monkeypatch.setattr('src.vc.voice_library.metadata', unreadable)
    window._update_voice_info()
    assert 'profile.json' in window.voice_hint.text()

    monkeypatch.setattr('src.vc.voice_library.metadata',
                        lambda *args, **kwargs: {'name': 'テストの声', 'reference_seconds': 7.5})
    window._update_voice_info()
    assert '7.5' in window.voice_hint.text()


def test_standard_voice_is_labelled_as_the_app_voice(tmp_path):
    """The shipped voice is never presented as one of the user's own registrations."""
    from src.vc import models
    from src.vc.voice_library import DEFAULT_VOICE_ID, DEFAULT_VOICE_NAME, is_standard
    assert DEFAULT_VOICE_NAME.startswith('標準ボイス')
    assert is_standard(DEFAULT_VOICE_ID)
    assert not is_standard('user_'+'a'*32)
    profile = models.VOICE_PROFILES[DEFAULT_VOICE_ID]
    assert profile['name'] == DEFAULT_VOICE_NAME
    assert profile['standard'] is True
    # A user-registered voice keeps its own name and is never marked standard.
    folder = voice_files(tmp_path)
    found = discover(tmp_path)[folder.name]
    assert found['name'] == '試験の声' and found['standard'] is False


def test_metadata_cannot_override_backend_or_queue_policy(tmp_path):
    folder = voice_files(tmp_path)
    path = folder/'profile.json'
    info = json.loads(path.read_text())
    info.update(backend='beatrice_vst', folder='../../other', queue_chunks=999)
    path.write_text(json.dumps(info))
    found = discover(tmp_path)[folder.name]
    assert found['queue_chunks'] == 10
    assert found['folder'] == 'user_voices/'+folder.name


def test_reference_selection_excludes_silence_and_clipped_windows():
    rate = 16000
    clean = (.1*np.sin(2*np.pi*210*np.arange(5*rate)/rate)).astype(np.float32)
    audio = np.concatenate([np.zeros(5*rate), clean, np.ones(5*rate)])
    parts, info = select_segments(audio)
    assert len(parts) == 1
    # A short lead-in is kept before the speech so a consonant onset is not cut off,
    # so the offset lands at the 5s boundary rather than exactly on it.
    assert 4.9 <= info[0]['offset_seconds'] <= 5.1
    assert info[0]['clipping_ratio'] == 0
    # The retained part is the clean tone: not the leading silence, not the clipped tail.
    assert len(parts[0]) == pytest.approx(5*rate, abs=0.3*rate)
    assert 20*np.log10(float(np.sqrt(np.mean(parts[0]**2)))) == pytest.approx(-23.0, abs=0.5)
    for bad in (np.zeros(5*rate), np.ones(5*rate), clean[:2*rate], np.full(5*rate, np.nan)):
        with pytest.raises(ValueError):
            select_segments(bad)


def test_centroid_preserves_scale_and_rejects_invalid_vectors():
    vector = np.arange(256, dtype=np.float32)+1
    result, keep = centroid([vector, vector*2, vector*3])
    np.testing.assert_allclose(result/np.linalg.norm(result), vector/np.linalg.norm(vector), rtol=1e-5)
    assert np.linalg.norm(result) == pytest.approx(np.median([np.linalg.norm(vector*(i+1)) for i in keep]))
    with pytest.raises(ValueError):
        centroid(np.zeros((3, 256), dtype=np.float32))


def test_registration_dialog_stops_worker_and_cancel_preserves_current_voice(ai_window):
    app, window, backend, bridge = ai_window
    window.mode.setCurrentIndex(window.mode.findData('ai_voice'))
    pump(app, lambda: bridge.status == 'Ready')
    selected = window.ai_model.currentData()
    window.add_voice.click()
    pump(app, lambda: not bridge.alive and not window._pending)
    dialog = window.voice_dialog
    assert dialog is not None and dialog.ready()
    assert not window.start_button.isEnabled()
    dialog.reject()
    pump(app, lambda: window.voice_dialog is None)
    assert window.ai_model.currentData() == selected
    assert window.start_button.isEnabled()


def test_registration_selects_the_new_voice(ai_window, monkeypatch, tmp_path):
    from src.vc import models
    from src.vc import voice_library
    from src.settings.manager import AppSettings
    from src.vc.models import DEFAULT_VOICE_ID
    app, window, backend, bridge = ai_window
    folder = voice_files(tmp_path)
    user_profiles = discover(tmp_path)
    monkeypatch.setattr(models, 'discover_user_voices', lambda: user_profiles)
    monkeypatch.setattr(voice_library, 'metadata', lambda identifier: json.loads((folder/'profile.json').read_text()))
    try:
        window.add_voice.click()
        pump(app, lambda: not window._pending)
        window.voice_dialog.registered_id = folder.name
        window.voice_dialog.accept()
        pump(app, lambda: not window._pending)
        # The freshly registered voice is offered and selected.
        assert folder.name in user_profiles
        assert window.ai_model.findData(folder.name) >= 0
        assert window.ai_model.currentData() == folder.name
        pump(app, lambda: bridge.parameters.model == folder.name)
        assert AppSettings.from_dict(dict(ai_model=folder.name)).ai_model == folder.name
        window._capture_settings()
        assert window.settings.ai_model == folder.name
    finally:
        monkeypatch.undo()
        models.refresh_user_profiles()
        # Leave the window on a real voice so teardown can capture settings.
        window.ai_model.blockSignals(True)
        window.ai_model.clear()
        for key, value in models.VOICE_PROFILES.items():
            window.ai_model.addItem(value['name'], key)
        window.ai_model.setCurrentIndex(window.ai_model.findData(DEFAULT_VOICE_ID))
        window.ai_model.blockSignals(False)
        assert models.default_voice_id() is not None


def test_registration_dialog_fits_on_screen_with_collapsible_sections():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea
    from src.gui import voice_registration as gui
    app = QApplication.instance() or QApplication([])
    dialog = gui.VoiceRegistrationDialog()
    try:
        # Three collapsible pages; clip/report pages unlock after phase 1.
        assert dialog.toolbox.count() == 3
        assert not dialog.toolbox.isItemEnabled(1)
        assert not dialog.toolbox.isItemEnabled(2)
        # The confirm button lives outside the scroll area, so a long clip list
        # can never push it off-screen or out of the tab order.
        scroll = dialog.findChild(QScrollArea)
        assert dialog.submit.parent() is not dialog.toolbox
        assert scroll is not None and dialog.submit not in scroll.findChildren(QPushButton)
        assert dialog.submit.focusPolicy() != Qt.FocusPolicy.NoFocus
        # Bulk toggles flip every clip checkbox at once.
        dialog.bundle_report = {'selections': [
            {'offset_seconds': 0.0, 'duration_seconds': 5.0},
            {'offset_seconds': 7.0, 'duration_seconds': 5.0}]}
        dialog.bundle = 'dummy'
        dialog.build_clip_rows()
        assert len(dialog.clips) == 2
        dialog.set_all_clips(False)
        assert dialog.selected_indices() == []
        dialog.set_all_clips(True)
        assert dialog.selected_indices() == [0, 1]
    finally:
        dialog.deleteLater()


def test_registration_worker_protocol_reports_unicode_and_recovers_from_failure(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from src.gui import voice_registration as gui
    import psutil
    app = QApplication.instance() or QApplication([])
    tools = tmp_path/'tools'
    tools.mkdir()
    identifier = 'user_'+'b'*32
    # Phase 1 reports a selection and leaves a bundle; the dialog must not publish yet.
    # Paths are embedded as JSON literals: a Windows path contains backslashes that would
    # otherwise be read as escapes in the generated source.
    bundle = str(tmp_path/('.pending_'+identifier))
    script = ('import json,sys\n'
              'if "--finalize" in sys.argv:\n'
              f'    print(json.dumps({{"registered":{json.dumps(identifier)}}}),flush=True)\n'
              'else:\n'
              '    print(json.dumps({"message":"声の特徴を抽出中 1/2…","report":'
              '{"retained_seconds":9.0,"selections":[{"offset_seconds":1.0,"duration_seconds":4.5,'
              '"quality":0.5,"snr_db":20.0,"f0_median_hz":180.0}]}}),flush=True)\n'
              f'    print(json.dumps({{"bundle":{json.dumps(bundle)},'
              f'"identifier":{json.dumps(identifier)}}}),flush=True)\n')
    (tools/'register_voice.py').write_text(script, 'utf-8')
    monkeypatch.setattr(gui, 'ROOT', tmp_path)
    monkeypatch.setattr(gui, 'data_dir', lambda: tmp_path)
    monkeypatch.setattr(gui, 'worker_python', lambda *_: (sys.executable, os.environ.copy()))
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=16*1024**3))
    dialog = gui.VoiceRegistrationDialog()
    dialog.source = str(tmp_path/'日本語の声.m4a')
    dialog.set_sources([dialog.source])
    dialog.name.setText('試験の声')
    dialog.open()
    dialog.on_submit()
    pump(app, lambda: dialog.process is None)
    # Phase 1 leaves the voice unpublished and offers one selectable clip.
    assert dialog.registered_id is None
    assert dialog.bundle == bundle
    assert len(dialog.clips) == 1 and dialog.clips[0].isChecked()
    # Phase 2 publishes only what the person kept.
    dialog.on_submit()
    pump(app, lambda: dialog.process is None)
    assert dialog.registered_id == identifier
    assert dialog.result() == dialog.DialogCode.Accepted
    assert not dialog.watchdog.isActive() and dialog.log_file is None
    dialog.deleteLater()


def test_registration_reports_a_failure_and_keeps_the_dialog_usable(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from src.gui import voice_registration as gui
    import psutil
    app = QApplication.instance() or QApplication([])
    tools = tmp_path/'tools'
    tools.mkdir()
    script = ('import json,sys\n'
              'print(json.dumps({"message":"声の特徴を抽出中…"}, ensure_ascii=False),flush=True)\n'
              'print(json.dumps({"error":"音割れが多い音声です"},ensure_ascii=False),flush=True)\nsys.exit(1)\n')
    (tools/'register_voice.py').write_text(script, 'utf-8')
    monkeypatch.setattr(gui, 'ROOT', tmp_path)
    monkeypatch.setattr(gui, 'data_dir', lambda: tmp_path)
    monkeypatch.setattr(gui, 'worker_python', lambda *_: (sys.executable, os.environ.copy()))
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=16*1024**3))
    dialog = gui.VoiceRegistrationDialog()
    dialog.set_sources([str(tmp_path/'声.wav')])
    dialog.name.setText('試験の声')
    dialog.open()
    dialog.on_submit()
    pump(app, lambda: dialog.process is None)
    assert dialog.submit.isEnabled()
    assert '音割れ' in dialog.status.text()
    assert dialog.registered_id is None
    assert dialog.bundle == ''
    dialog.reject()
    dialog.deleteLater()


def test_unpublished_bundle_is_removed_when_the_dialog_is_cancelled(tmp_path):
    from PySide6.QtWidgets import QApplication
    from src.gui import voice_registration as gui
    app = QApplication.instance() or QApplication([])
    library = tmp_path/'models/user_voices'
    library.mkdir(parents=True)
    bundle = library/('.pending_user_'+'c'*32)
    bundle.mkdir()
    dialog = gui.VoiceRegistrationDialog()
    dialog.bundle = str(bundle)
    dialog.reject()
    assert not bundle.exists()
    assert dialog.result() == dialog.DialogCode.Rejected
    dialog.deleteLater()


def test_registration_worker_script_is_found_in_the_installed_tree(tmp_path, monkeypatch):
    """A packaged GUI bundles no ``tools/``, so the worker comes from the installed tree.

    The bundle's own root is still tried first: unfrozen, and in any build that does
    ship the script, that is the tree the library lives in.
    """
    from src.gui import voice_registration as gui
    bundle = tmp_path/'_internal'
    bundle.mkdir()
    installed = tmp_path/'Koeiro'
    (installed/'tools').mkdir(parents=True)
    script = installed/'tools/register_voice.py'
    script.write_text('# fixture', 'utf-8')
    monkeypatch.setattr(gui, 'ROOT', bundle)
    monkeypatch.setattr(gui, 'install_root', lambda: installed)
    assert gui.worker_script() == script
    # A bundle that does carry the script keeps using it.
    (bundle/'tools').mkdir()
    shipped = bundle/'tools/register_voice.py'
    shipped.write_text('# fixture', 'utf-8')
    assert gui.worker_script() == shipped
    # Neither tree has one: the dialog must say so instead of launching a missing file
    # and reporting a generic failure.
    monkeypatch.setattr(gui, 'ROOT', tmp_path/'nowhere')
    monkeypatch.setattr(gui, 'install_root', lambda: tmp_path/'also-nowhere')
    assert gui.worker_script() is None


def test_dialog_stops_before_launching_a_missing_worker(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from src.gui import voice_registration as gui
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui, 'ROOT', tmp_path)
    monkeypatch.setattr(gui, 'install_root', lambda: tmp_path)
    monkeypatch.setattr(gui, 'worker_python',
                        lambda *_: (_ for _ in ()).throw(AssertionError('must not launch')))
    dialog = gui.VoiceRegistrationDialog()
    try:
        dialog.set_sources([str(tmp_path/'声.wav')])
        dialog.name.setText('試験の声')
        dialog.on_submit()
        assert dialog.process is None
        assert '見つかりません' in dialog.status.text()
    finally:
        dialog.deleteLater()


def test_voice_library_follows_the_installed_tree_when_packaged(tmp_path, monkeypatch):
    """Registration writes where the AI worker reads, so the listing has to follow it."""
    from src.vc import voice_library
    bundle, installed = tmp_path/'_internal', tmp_path/'tree'
    (installed/'models'/'user_voices').mkdir(parents=True)
    monkeypatch.setattr(voice_library, 'is_frozen', lambda: True)
    monkeypatch.setattr(voice_library, 'asset_root', lambda: bundle)
    monkeypatch.setattr(voice_library, 'install_root', lambda: installed)
    assert voice_library.library_root() == installed
    # A standalone bundle with no installed tree keeps reading its own copy, so the
    # shipped voice is still offered.
    shutil.rmtree(installed/'models'/'user_voices')
    assert voice_library.library_root() == bundle
    monkeypatch.setattr(voice_library, 'is_frozen', lambda: False)
    assert voice_library.library_root() == bundle


def test_selected_voice_button_activates_ai_without_changing_voice(ai_window):
    app, window, backend, bridge = ai_window
    # The button exists to switch the app onto the AI route, so it is offered exactly
    # when that route is not active yet. It lives in the voice drawer section.
    window._open_drawer('tune')
    pump(app, lambda: window.drawer_tune.isVisible())
    window.mode.setCurrentIndex(window.mode.findData('original'))
    pump(app, lambda: window.use_voice.isVisible())
    selected = window.ai_model.currentData()
    window.use_voice.click()
    pump(app, lambda: bridge.status == 'Ready')
    assert window.mode.currentData() == 'ai_voice'
    assert window.ai_model.currentData() == selected
    pump(app, lambda: not window.use_voice.isVisible())


def test_add_to_destination_passes_the_voice_identifier():
    """The registration worker resolves a folder id, so a label must never be sent."""
    from PySide6.QtWidgets import QApplication
    from src.gui import voice_registration as gui
    QApplication.instance() or QApplication([])
    identifier = 'user_'+'b'*32
    dialog = gui.VoiceRegistrationDialog(ready=lambda: False)
    try:
        dialog.set_voices([('登録した声', identifier)])
        assert dialog.voice_picker.itemText(0) == '登録した声'
        assert dialog.voice_picker.itemData(0) == identifier
        dialog.add_to.setChecked(True)
        dialog.begin()
        assert dialog.target_voice == identifier
        arguments = dialog.build_arguments()
        assert arguments[arguments.index('--add-to')+1] == identifier
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_watchdog_decision_flags_stalls_and_the_absolute_cap_separately():
    from src.gui import voice_registration as gui
    assert gui.watchdog_decision(100.0, 0.0, 95.0) is None
    assert gui.watchdog_decision(100.0, 0.0, 100.0-gui.WATCHDOG_STALL_SECONDS-1.0) == 'stalled'
    assert gui.watchdog_decision(gui.WATCHDOG_ABSOLUTE_SECONDS+1.0, 0.0,
                                 gui.WATCHDOG_ABSOLUTE_SECONDS+1.0) == 'timeout'


def test_registration_watchdog_only_kills_a_silent_worker(tmp_path, monkeypatch):
    """Registration time is not the failure signal: an hour in, a worker that is still
    printing progress must survive, and only silence past the stall window is stopped."""
    from PySide6.QtWidgets import QApplication
    from src.gui import voice_registration as gui
    app = QApplication.instance() or QApplication([])
    tools = tmp_path/'tools'
    tools.mkdir()
    script = ('import json,sys,time\n'
              'print(json.dumps({"message":"登録を開始しました"}, ensure_ascii=False),flush=True)\n'
              'time.sleep(30)\n')
    (tools/'register_voice.py').write_text(script, 'utf-8')
    monkeypatch.setattr(gui, 'ROOT', tmp_path)
    monkeypatch.setattr(gui, 'data_dir', lambda: tmp_path)
    monkeypatch.setattr(gui, 'worker_python', lambda *_: (sys.executable, os.environ.copy()))
    dialog = gui.VoiceRegistrationDialog()
    dialog.set_sources([str(tmp_path/'声.wav')])
    dialog.name.setText('試験の声')
    dialog.open()
    dialog.on_submit()
    try:
        pump(app, lambda: '開始' in dialog.status.text())
        dialog.started = time.monotonic()-3600.0
        dialog.last_activity = time.monotonic()
        dialog.check_resources()
        assert dialog.process is not None and dialog.process.processId()
        dialog.last_activity = time.monotonic()-(gui.WATCHDOG_STALL_SECONDS+5.0)
        dialog.check_resources()
        pump(app, lambda: dialog.process is None)
        assert '応答' in dialog.status.text()
    finally:
        if dialog.process is not None:
            dialog.reject()
            pump(app, lambda: dialog.process is None)
        dialog.deleteLater()
