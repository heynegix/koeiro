import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_ai_gui import ai_window, pump  # noqa: F401  (ai_window is used as a fixture)
from tests.test_voice_quality import RATE, speech, write_wav  # noqa: F401  (RATE used below)


# --- shell widgets ---

@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_level_bar_clamps_and_never_shows_signal_it_was_not_given(app):
    from src.gui.shell import LevelBar
    bar = LevelBar(bars=8)
    bar.set_level(0.5)
    assert 0.0 <= bar._peak <= 1.0
    bar.set_level(9.0)
    assert bar._peak == 1.0
    bar.set_level(-3.0)
    assert bar._peak >= 0.0
    # A silent bar must read as silent.
    bar2 = LevelBar(bars=8)
    bar2.set_level(0.0)
    assert bar2._peak == 0.0


def test_orb_reports_state_and_is_tappable(app):
    from src.gui.shell import MicOrb
    orb = MicOrb()
    orb.set_state('マイク入力中', '声が出力されます', active=True)
    assert orb._state == 'マイク入力中'
    assert orb._active is True
    orb.set_level(0.0)
    assert orb._level == 0.0
    fired = []
    orb.tapped.connect(lambda: fired.append(True))
    orb.resize(260, 260)
    orb.grab()
    orb.tapped.emit()
    assert fired == [True]


def test_sidebar_switches_pages_and_rejects_unknown_keys(app):
    from src.gui.shell import Sidebar
    sidebar = Sidebar()
    seen = []
    sidebar.changed.connect(seen.append)
    sidebar.select('library')
    assert seen == ['library']
    assert sidebar.buttons['library'].isChecked()
    sidebar.select('nonexistent')
    assert seen == ['library']


def test_preset_rail_marks_exactly_one_card(app):
    from src.gui.shell import PresetRail
    rail = PresetRail((('a', 'A', 'detail', '♫'), ('b', 'B', 'detail', '♂')))
    chosen = []
    rail.selected.connect(chosen.append)
    rail.choose('b')
    assert chosen == ['b']
    assert rail.cards['b'].property('selected') is True
    assert rail.cards['a'].property('selected') is False
    rail.select_silently('a')
    assert rail.cards['a'].property('selected') is True


# --- preview module ---

def test_preview_reads_and_writes_wav_without_changing_the_signal(tmp_path):
    from src.gui import preview
    source = write_wav(tmp_path/'in.wav', speech(3.0).astype(np.float32))
    audio = preview.read_wav(source)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == pytest.approx(3.0*48000, rel=0.01)
    target = preview.write_wav(tmp_path/'nested'/'out.wav', audio)
    assert target.is_file()
    again = preview.read_wav(target)
    # 16-bit round trip: close, not bit exact.
    assert np.max(np.abs(again-audio)) < 2.0/32768


def test_preview_rejects_unreadable_and_silent_input(tmp_path):
    from src.gui import preview
    broken = tmp_path/'broken.wav'
    broken.write_bytes(b'not a wav file at all')
    with pytest.raises(preview.PreviewError):
        preview.read_wav(broken)
    silence = write_wav(tmp_path/'silence.wav', np.zeros(int(2*48000), np.float32))
    with pytest.raises(preview.PreviewError, match='無音'):
        preview.convert(silence, 'meanvc2_ref20')
    with pytest.raises(preview.PreviewError):
        preview.convert(tmp_path/'missing.wav', 'meanvc2_ref20')


def test_preview_cache_key_separates_every_condition():
    """A LavaSR preview must never be served a plain conversion from the cache."""
    import inspect
    from src.gui import preview
    source = inspect.getsource(preview.convert)
    for token in ('str(model)', 'str(enhancer)', 'str(bool(denoise))', 'str(experiment)',
                  'str(tune_sib_db)', 'str(tune_cons_db)', 'str(tune_caps)',
                  'str(tune_floor_db)', 'str(tune_excess_db)', 'str(tune_mid)',
                  'str(tune_match)', 'str(tune_ptrans)', 'str(tune_pcap)',
                  'str(bool(tune_combined))', 'str(tune_level_db)',
                  'PREVIEW_FORMAT_VERSION'):
        assert token in source, f'cache key ignores {token}'


def test_preview_describe_surfaces_the_worker_measurements():
    from src.gui import preview
    text = preview.describe({'rtf': 0.371, 'guards': {'input_silence_seconds': 1.5,
                                                      'enhancer_skipped': True,
                                                      'enhancer_skip_reason': 'entirely silent'},
                             'enhancer': 'lavasr'})
    assert '0.371' in text
    assert '1.5' in text
    assert 'entirely silent' in text
    assert 'lavasr' in text
    assert preview.describe({}) == ''


# --- window integration ---

def test_window_rail_preset_switches_delivery(ai_window):
    app, window, _backend, bridge = ai_window
    window.sidebar.select('home')
    pump(app, lambda: window._active_page == 'home')
    window._apply_rail_preset('streaming')
    pump(app, lambda: not window._pending)
    assert window.mode.currentData() == 'ai_voice'
    assert window.ai_delivery.currentData() == 'streaming'
    assert bridge.parameters.delivery == 'streaming'
    window._apply_rail_preset('utterance_lavasr')
    pump(app, lambda: not window._pending)
    assert window.ai_delivery.currentData() == 'utterance_lavasr'
    assert bridge.parameters.delivery == 'utterance' and bridge.parameters.enhancer == 'lavasr'
    assert '逐次変換' in window.rail_note.text() or '一括変換' in window.rail_note.text()


def test_window_rail_preset_key_that_no_longer_ships_is_ignored(ai_window):
    app, window, _backend, _bridge = ai_window
    window._apply_rail_preset('female_high')
    pump(app, lambda: not window._pending)
    assert window.ai_delivery.currentData() != 'female_high'


def test_window_rail_preset_is_refused_while_running(ai_window):
    app, window, _backend, _bridge = ai_window
    window._pending = True
    window._apply_rail_preset('streaming')
    assert 'Stop' in window.rail_note.text()
    window._pending = False


def test_window_pages_switch_and_hide_the_others(ai_window):
    app, window, _backend, _bridge = ai_window
    for key in ('presets', 'library', 'advanced', 'settings', 'home'):
        window.sidebar.select(key)
        pump(app, lambda name=key: window._active_page == name)
        visible = [name for name, (widget, _l) in window.pages.items() if widget.isVisible()]
        assert visible == [key]


def test_window_pitch_slider_is_only_live_on_the_dsp_path(ai_window):
    app, window, _backend, _bridge = ai_window
    window.mode.setCurrentIndex(window.mode.findData('female_dsp'))
    pump(app, lambda: window.pitch_display.isEnabled())
    assert window.pitch_display.isEnabled()
    window.mode.setCurrentIndex(window.mode.findData('ai_voice'))
    pump(app, lambda: not window.pitch_display.isEnabled())
    assert not window.pitch_display.isEnabled()


def test_window_orb_shows_stopped_without_inventing_signal(ai_window):
    app, window, _backend, _bridge = ai_window
    pump(app, lambda: window.orb is not None)
    window._tick()
    assert window.orb._level == 0.0
    assert '停止' in window.orb._state


def test_window_preview_button_requires_ai_voice(ai_window):
    app, window, _backend, _bridge = ai_window
    window.mode.setCurrentIndex(window.mode.findData('ai_voice'))
    pump(app, lambda: window.preview_button.isEnabled())
    assert window.preview_button.isEnabled()
    window.mode.setCurrentIndex(window.mode.findData('original'))
    pump(app, lambda: not window.preview_button.isEnabled())
    assert not window.preview_button.isEnabled()


def test_window_preview_refuses_while_the_engine_runs(ai_window):
    app, window, _backend, _bridge = ai_window
    window._preview_busy = True
    window._preview_converted()
    assert '試聴できません' in window.preview_note.text() or window._preview_busy


def test_window_stopping_preview_resets_the_button(ai_window):
    app, window, _backend, _bridge = ai_window
    window.preview_button.setText('再生を停止')
    window._stop_preview()
    assert window.preview_button.text() == '変換を試聴'


def test_window_output_card_describes_the_active_mode(ai_window):
    app, window, _backend, _bridge = ai_window
    window.mode.setCurrentIndex(window.mode.findData('original'))
    assert window._describe_output() == 'Original（無変換）'
    window.mode.setCurrentIndex(window.mode.findData('female_dsp'))
    assert window._describe_output() == 'Female DSP'
    window.mode.setCurrentIndex(window.mode.findData('ai_voice'))
    described = window._describe_output()
    assert window.ai_model.currentText() in described


# --- runtime guards surfaced through the worker stats ---

def test_utterance_statistics_report_silence_and_skips():
    from src.vc.utterance import convert_utterance

    class FakeBackend:
        chunk_samples = 2560
        sample_rate = 16000

        def __init__(self):
            self.calls = 0

        def reset(self):
            pass

        def get_stats(self):
            return {'algorithmic_buffer_ms': 1440, 'interpolation_grid_delay_ms': 0,
                    'phrase_extra_delay_ms': 1600}

        def process_chunk(self, audio):
            self.calls += 1
            return audio

    backend = FakeBackend()
    # convert_utterance runs at 48 kHz, so the fixture is generated at that rate.
    gap = np.zeros(int(1.0*48000), np.float32)
    audio = np.concatenate([speech(2.0, rate=48000).astype(np.float32), gap,
                            speech(2.0, rate=48000).astype(np.float32)]).astype(np.float32)
    stats = {}
    result = convert_utterance(backend, None, None, audio, statistics=stats)
    assert len(result) == len(audio)
    assert stats['input_seconds'] == pytest.approx(5.0, abs=0.05)
    assert stats['input_silence_seconds'] > 0.5
    assert 0.0 < stats['silence_share'] < 1.0
    # Without an enhancer there is nothing to skip, so no enhancer key is written.
    assert 'enhancer_skipped' not in stats


def test_utterance_skips_an_enhancer_when_the_result_is_silent():
    from src.vc.utterance import convert_utterance

    class SilentBackend:
        chunk_samples = 2560
        sample_rate = 16000

        def __init__(self):
            self.calls = 0

        def reset(self):
            pass

        def get_stats(self):
            return {'algorithmic_buffer_ms': 1440, 'interpolation_grid_delay_ms': 0}

        def process_chunk(self, audio):
            self.calls += 1
            return np.zeros_like(audio)

    ran = []

    class RecordingEnhancer:
        def process(self, audio):
            ran.append(len(audio))
            return audio

    backend = SilentBackend()
    audio = np.zeros(int(3.0*48000), np.float32)
    stats = {}
    result = convert_utterance(backend, None, None, audio,
                               enhance=RecordingEnhancer().process, statistics=stats)
    assert ran == [], 'the enhancer ran on an inaudible result'
    assert stats['enhancer_skipped'] is True
    assert len(result) == len(audio)


def test_utterance_reports_an_enhancer_failure_rather_than_hiding_it():
    from src.vc.utterance import convert_utterance

    class SimpleBackend:
        chunk_samples = 2560
        sample_rate = 16000

        def __init__(self):
            pass

        def reset(self):
            pass

        def get_stats(self):
            return {'algorithmic_buffer_ms': 1440, 'interpolation_grid_delay_ms': 0}

        def process_chunk(self, audio):
            return audio

    def broken(audio):
        raise RuntimeError('enhancer exploded')

    audio = speech(3.0, rate=48000).astype(np.float32)
    with pytest.raises(RuntimeError, match='enhancer exploded'):
        convert_utterance(SimpleBackend(), None, None, audio, enhance=broken, statistics={})


# --- listening A/B harness ---

def test_ab_harness_normalises_level_between_conditions():
    from tools.build_listening_ab import normalise
    quiet = (speech(2.0, level=0.01).astype(np.float32))
    loud = (speech(2.0, level=0.20).astype(np.float32))
    a = normalise(quiet, 0.05)
    b = normalise(loud, 0.05)
    def rms(x):
        return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))
    assert rms(a) == pytest.approx(rms(b), rel=0.02)
    # A silent clip is returned untouched rather than scaled into noise.
    assert normalise(np.zeros(1000, np.float32), 0.05).max() == 0.0


def test_ab_harness_builds_a_shuffled_pair_with_a_sealed_answer(tmp_path):
    from tools.build_listening_ab import build_pair, measure
    import numpy as np
    source = speech(3.0, rate=48000).astype(np.float32)
    rng = np.random.default_rng(0)

    def convert(audio, rate, variant=0.0, **_ignored):
        return audio*(1.0+variant)

    result = build_pair(source, 48000, convert,
                        {'no change': dict(variant=0.0), 'scaled': dict(variant=0.2)},
                        tmp_path, rng)
    assert result is not None
    assert len(result['pages']) == 2
    assert set(result['answer']) == {'A', 'B'}
    # Both sides must be different conditions, or the pair proves nothing.
    assert result['answer']['A'] != result['answer']['B']
    for page in result['pages']:
        assert (tmp_path/page['file']).is_file()
    assert 'spectral_flatness' in measure(source, 48000)


def test_ab_harness_skips_a_pair_when_a_condition_fails(tmp_path):
    from tools.build_listening_ab import build_pair
    source = speech(3.0, rate=48000).astype(np.float32)

    def convert(audio, rate, fail=False, **_ignored):
        return None if fail else audio

    assert build_pair(source, 48000, convert,
                      {'ok': dict(), 'broken': dict(fail=True)}, tmp_path,
                      np.random.default_rng(1)) is None