import threading
from concurrent.futures import Future

import numpy as np
import pytest
from tests.test_ai_gui import ai_window

from src.vc.phrase_prosody import HOP, HISTORY, LOOKAHEAD, PhraseRepair, pitch_edit, repair_window
from src.vc.phrase_prosody import PhraseAnalyzer
from src.vc.config import AIParameters
from src.vc.models import DEFAULT_VOICE_ID, profile
from src.settings.manager import AppSettings


def test_silence_is_exact_and_has_no_false_pitch():
    x = np.zeros(HISTORY + LOOKAHEAD, np.float32)
    out, stats = repair_window(x, x, HISTORY)
    assert not out.any() and stats['voiced_frames'] == stats['pitch_regions'] == 0


def test_identical_contours_are_not_modified():
    t = np.arange(HISTORY + LOOKAHEAD) / 16000
    x = (.15*np.sin(2*np.pi*220*t)).astype(np.float32)
    out, stats = repair_window(x, x, HISTORY)
    np.testing.assert_array_equal(out, x[HISTORY:HISTORY+HOP])
    assert stats['max_shift_st'] == stats['max_gain_db'] == 0


def test_psola_keeps_duration_and_identity_and_can_change_pitch():
    t = np.arange(16000) / 16000
    x = (.15*np.sin(2*np.pi*220*t)).astype(np.float32)
    np.testing.assert_array_equal(pitch_edit(x, 220, 0), x)
    out = pitch_edit(x, 220, .6)
    assert out.shape == x.shape and np.isfinite(out).all()
    region = out[2000:-2000]
    spectrum = abs(np.fft.rfft(region*np.hanning(len(region)), n=65536))
    frequency = np.argmax(spectrum)*16000/65536
    assert 225 < frequency < 232
    assert np.max(abs(out)) < .2


def test_absolute_male_pitch_is_not_copied_to_voice():
    t = np.arange(HISTORY + LOOKAHEAD) / 16000
    source = (.2*np.sin(2*np.pi*110*t)).astype(np.float32)
    voice = (.1*np.sin(2*np.pi*220*t)).astype(np.float32)
    out, stats = repair_window(source, voice, HISTORY)
    assert stats['max_shift_st'] < .03 and stats['max_gain_db'] < .03
    assert np.max(abs(out-voice[HISTORY:HISTORY+HOP])) < .0002


def test_relative_energy_repair_is_bounded_and_does_not_shift_pitch():
    t=np.arange(HISTORY+LOOKAHEAD)/16000
    source=((.1+.07*np.sin(2*np.pi*.6*t))*np.sin(2*np.pi*110*t)).astype(np.float32)
    voice=(.1*np.sin(2*np.pi*220*t)).astype(np.float32)
    out,stats=repair_window(source,voice,HISTORY,energy=True,pitch=False)
    assert 0 < stats['max_gain_db'] <= 1.5 and stats['pitch_regions']==0
    assert np.isfinite(out).all() and len(out)==HOP
    assert np.sqrt(np.mean((out-voice[HISTORY:HISTORY+HOP])**2)) > .0001


def test_relative_pitch_repair_is_small_and_preserves_output_clock():
    t=np.arange(HISTORY+LOOKAHEAD)/16000
    f0=110*2**(1.5*np.sin(2*np.pi*.7*t)/12)
    source=(.1*np.sin(2*np.pi*np.cumsum(f0)/16000)).astype(np.float32)
    voice=(.1*np.sin(2*np.pi*220*t)).astype(np.float32)
    out,stats=repair_window(source,voice,HISTORY,energy=False,pitch=True)
    assert 0 < stats['max_shift_st'] <= .6 and stats['max_gain_db']==0
    assert stats['pitch_regions']>0 and out.shape==(HOP,) and np.isfinite(out).all()


def test_delay_and_bounded_memory_are_independent_of_analysis_deadline():
    p = PhraseRepair(0)
    never = Future()
    p.pending = (p.epoch, 0, never)
    try:
        for i in range(70):
            x = np.full(HOP, i/100, np.float32)
            out = p.process(x, x)
            np.testing.assert_array_equal(out, np.full(HOP, max(0, i-10)/100, np.float32))
            assert len(p.voice) <= HISTORY+LOOKAHEAD+HOP
        assert p.stats['late_bypass'] == 60 and p.stats['busy_skip'] > 0
    finally:
        never.cancel()
        p.close()


def test_reset_rejects_a_previous_epochs_completed_edit():
    p = PhraseRepair(0)
    old = p.epoch
    completed = Future()
    completed.set_result((np.ones(HOP, np.float32), {}))
    p.pending = (old, 0, completed)
    p.reset()
    try:
        assert not p.process(np.zeros(HOP, np.float32), np.zeros(HOP, np.float32)).any()
        assert not p.completed and p.stats['applied'] == 0
    finally:
        p.close()


def test_analysis_error_bypasses_voice_and_is_reported():
    p = PhraseRepair(0)
    for _ in range(10):
        p.end += HOP
        p.voice = np.concatenate((p.voice, np.full(HOP, .2, np.float32)))
        p.source = p.voice.copy()
    failed = Future()
    failed.set_exception(RuntimeError('analysis failure'))
    p.pending = (p.epoch, 0, failed)
    try:
        out = p.process(np.full(HOP, .2, np.float32), np.full(HOP, .2, np.float32))
        np.testing.assert_array_equal(out, np.full(HOP, .2, np.float32))
        assert p.stats['failures'] == 1
        assert p.stats['disabled'] and p.stats['last_error']=='analysis failure'
    finally:
        p.close()


def test_cached_analysis_has_same_waveform_and_only_analyzes_new_frames():
    size=HISTORY+LOOKAHEAD
    t=np.arange(size+HOP)/16000
    x=(.15*np.sin(2*np.pi*220*t)).astype(np.float32)
    analyzer=PhraseAnalyzer()
    _,first=analyzer(x[:size],x[:size],HISTORY,True,True,0,1)
    out,stats=analyzer(x[HOP:],x[HOP:],HISTORY,True,True,HOP,1)
    expected,_=repair_window(x[HOP:],x[HOP:],HISTORY)
    np.testing.assert_array_equal(out,expected)
    assert stats['estimated_frames']==8 and stats['estimated_frames']<first['estimated_frames']
    assert stats['cached_frames']==first['cached_frames']
    _,reset=analyzer(x[HOP:],x[HOP:],HISTORY,True,True,HOP,2)
    assert reset['estimated_frames']==first['estimated_frames']


def test_app_profile_roundtrip_and_callback_queue_limits():
    # The shipped voice carries phrase repair, so it reaches the worker through the
    # normal profile path rather than a separately named variant.
    params = AppSettings.from_dict({'ai_pitch':8,'ai_post_fx':True}).ai_parameters()
    assert params.model == DEFAULT_VOICE_ID and params.pitch == 0 and not params.post_fx
    assert params.queue_chunks == 10 and params.chunk_frames == 7680
    assert params.mute_during_startup
    assert profile(DEFAULT_VOICE_ID).get('phrase_repair')


def test_invalid_audio_is_rejected_before_scheduling():
    p = PhraseRepair(0)
    try:
        with pytest.raises(ValueError):
            p.process(np.full(HOP, np.nan, np.float32), np.zeros(HOP, np.float32))
        assert p.end == 0 and p.pending is None
    finally:
        p.close()


@pytest.mark.parametrize('alignment', [15360, 26240, 32000])
def test_grouped_voice_alignment_is_exact_and_stays_bounded(alignment):
    repair = PhraseRepair(alignment)
    never = Future()
    repair.pending = (repair.epoch, 0, never)
    source = np.arange(35*HOP, dtype=np.float32)/(35*HOP)
    expected = np.concatenate((np.zeros(alignment, np.float32), source))
    try:
        for offset in range(0, len(source), HOP):
            chunk = source[offset:offset+HOP]
            repair.process(chunk, chunk)
            np.testing.assert_array_equal(repair.source[-HOP:], expected[offset:offset+HOP])
            assert len(repair.source_delay) == alignment
            assert len(repair.source) <= HISTORY+LOOKAHEAD+HOP
    finally:
        never.cancel()
        repair.close()


@pytest.mark.parametrize('alignment', [-1, 32001, 1.5, True])
def test_alignment_limit_still_rejects_invalid_or_unbounded_buffers(alignment):
    with pytest.raises(ValueError, match='Invalid neural/source alignment'):
        PhraseRepair(alignment)


# Formerly read from models/meanvc2_ref20|ref60/runtime.json; those legacy dev
# profiles are no longer shipped (the app only offers user registrations), so
# the two real configurations are pinned here instead of on disk.
@pytest.mark.parametrize('frontend, interpolation, groups, frames, expected_delay', [
    ('legacy', 'legacy', 1, 1, 960),
    ('aligned', 'fixed_linear', 6, 1, 1640),
])
def test_phrase_backend_accepts_both_real_runtime_configurations(monkeypatch, frontend, interpolation, groups, frames, expected_delay):
    from src.vc.meanvc2 import MeanVC2Backend
    from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
    def load_config(backend, _path):
        backend.feature_frontend = frontend
        backend.bn_interpolation = interpolation
        backend.vc_group_chunks = groups
        backend.vocoder_batch_frames = frames
    monkeypatch.setattr(MeanVC2Backend, 'load', load_config)
    monkeypatch.setattr(MeanVC2Backend, 'reset', lambda _backend: None)
    backend = MeanVC2PhraseBackend()
    try:
        backend.load(None)
        assert backend.stats['phrase_source_alignment_ms'] == expected_delay
        assert backend.repair.alignment == expected_delay*16
        assert backend.stats['algorithmic_buffer_ms'] == expected_delay-(40 if interpolation == 'fixed_linear' else 0)
    finally:
        if backend.repair:
            backend.repair.close()


def test_gui_runs_repair_without_enabling_old_text_prosody(ai_window):
    from tests.test_ai_gui import pump
    app, window, backend, bridge = ai_window
    pump(app, lambda: bridge.parameters.model == DEFAULT_VOICE_ID and not window._pending)
    assert not window.prosody_on.isChecked()
    assert window.prosody_engine.currentData() == 'off'
    assert not window.ai_pitch.isEnabled() and not window.ai_post_fx.isEnabled()
    # The worker has not reported yet, so the extra buffering wait is not shown as if
    # it had already been applied.
    assert '1600ms' not in window.ai_performance.text()
