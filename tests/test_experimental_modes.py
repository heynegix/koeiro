"""Listening-comparison mode: same voice, every experimental improvement at once.

The two shipped routes (streaming, utterance+LavaSR) must behave exactly as
before; the integrated comparison mode never touches them.
"""
import numpy as np
import pytest

from src.vc.config import AIParameters, EXPERIMENTS
from src.vc.models import DELIVERY_MODES
from src.vc.phrase_prosody import (PhraseRepair, breath_sample_mask, retrospective_pitch,
                                   retrospective_repair, utterance_metrics)
from src.vc.utterance import (UtteranceCollector, blend_highs, clean_input, convert_utterance,
                              crossfade_blend, crossfade_join, detect_clicks, fade_edges,
                              fry_fraction, level_utterance, lift_consonants, split_at_pauses,
                              suppress_floor, tame_plosives, tame_sibilance)
from src.settings.manager import AppSettings
from tests.test_ai_gui import ai_window, pump  # noqa: F401 (ai_window is used as a fixture)


SHIPPED = ('streaming', 'utterance_lavasr')


def test_shipped_modes_carry_no_experiment():
    for key in SHIPPED:
        spec = DELIVERY_MODES[key]
        assert spec.get('experiment', 'none') == 'none'
        params = AIParameters(delivery=spec['delivery'], enhancer=spec['enhancer'])
        assert params.experiment == 'none'


def test_gui_selects_an_experimental_mode_and_returns(ai_window):
    app, window, backend, bridge = ai_window
    window.ai_delivery.setCurrentIndex(window.ai_delivery.findData('utterance_x_all'))
    pump(app, lambda: not window._pending)
    assert bridge.parameters.experiment == 'all'
    assert (bridge.parameters.delivery, bridge.parameters.enhancer) == ('utterance', 'lavasr')
    window._capture_settings()
    assert window.settings.ai_experiment == 'all'
    assert '比較用' in window.delivery_note.text()
    window.ai_delivery.setCurrentIndex(window.ai_delivery.findData('utterance_lavasr'))
    pump(app, lambda: not window._pending)
    assert bridge.parameters.experiment == 'none'
    window._capture_settings()
    assert window.settings.ai_experiment == 'none'


def test_tune_defaults_reproduce_validated_recipe():
    params = AIParameters(delivery='utterance', enhancer='lavasr', experiment='natural')
    assert (params.tune_sib_db, params.tune_cons_db, params.tune_caps,
            params.tune_floor_db, params.tune_excess_db) == (3.0, 3.0, 1.0, 3.0, 9.0)
    assert (params.tune_mid, params.tune_match, params.tune_ptrans,
            params.tune_pcap, params.tune_combined,
            params.tune_level_db) == (0.8, 0.25, 0.20, 1.0, True, -20.0)


def test_tune_rejects_out_of_range_values():
    bad = [dict(tune_sib_db=-0.1), dict(tune_sib_db=6.1), dict(tune_cons_db=99),
           dict(tune_caps=0.4), dict(tune_caps=1.6), dict(tune_floor_db=-1),
           dict(tune_excess_db=2.9), dict(tune_excess_db=24.1),
           dict(tune_caps=float('nan')), dict(tune_sib_db='loud'),
           dict(tune_mid=-0.1), dict(tune_mid=1.1), dict(tune_match=-0.1),
           dict(tune_match=0.6), dict(tune_ptrans=0.6), dict(tune_pcap=2.1),
           dict(tune_pcap=float('nan')), dict(tune_combined='yes'),
           dict(tune_level_db=-27), dict(tune_level_db=-13),
           dict(tune_level_db=float('inf'))]
    for changes in bad:
        with pytest.raises(ValueError):
            AIParameters(delivery='utterance', enhancer='lavasr',
                         experiment='natural', **changes)


def test_tune_settings_roundtrip_and_fallback():
    values = AppSettings.from_dict({'ai_tune_sib_db': 1.5, 'ai_tune_cons_db': 0.0,
                                    'ai_tune_caps': 1.25, 'ai_tune_floor_db': 2.0,
                                    'ai_tune_excess_db': 12.0, 'ai_tune_mid': 0.8,
                                    'ai_tune_match': 0.1, 'ai_tune_ptrans': 0.0,
                                    'ai_tune_pcap': 0.5, 'ai_tune_combined': False,
                                    'ai_tune_level_db': -22.5})
    params = values.ai_parameters()
    assert (params.tune_sib_db, params.tune_cons_db, params.tune_caps,
            params.tune_floor_db, params.tune_excess_db) == (1.5, 0.0, 1.25, 2.0, 12.0)
    assert (params.tune_mid, params.tune_match, params.tune_ptrans,
            params.tune_pcap, params.tune_combined,
            params.tune_level_db) == (0.8, 0.1, 0.0, 0.5, False, -22.5)
    broken = AppSettings.from_dict({'ai_tune_sib_db': 'x', 'ai_tune_caps': 9,
                                    'ai_tune_excess_db': float('nan'),
                                    'ai_tune_mid': None, 'ai_tune_combined': 'yes',
                                    'ai_tune_level_db': 0})
    assert (broken.ai_tune_sib_db, broken.ai_tune_caps,
            broken.ai_tune_excess_db) == (3.0, 1.0, 9.0)
    assert (broken.ai_tune_mid, broken.ai_tune_combined,
            broken.ai_tune_level_db) == (0.8, True, -20.0)


def test_gui_tune_sliders_drive_natural_only(ai_window):
    app, window, backend, bridge = ai_window
    window.ai_delivery.setCurrentIndex(window.ai_delivery.findData('utterance_x_natural'))
    pump(app, lambda: not window._pending)
    assert bridge.parameters.experiment == 'natural'
    for widget in window._tune_widgets:
        assert widget.isEnabled()
    window.tune_cons.setValue(10)
    pump(app, lambda: not window._pending)
    assert bridge.parameters.tune_cons_db == pytest.approx(1.0)
    window.tune_mid.setValue(80)
    pump(app, lambda: not window._pending)
    assert bridge.parameters.tune_mid == pytest.approx(0.8)
    window.tune_combined.setChecked(False)
    pump(app, lambda: not window._pending)
    assert bridge.parameters.tune_combined is False
    window._capture_settings()
    assert window.settings.ai_tune_cons_db == pytest.approx(1.0)
    assert window.settings.ai_tune_mid == pytest.approx(0.8)
    assert window.settings.ai_tune_combined is False
    window.tune_reset.click()
    pump(app, lambda: not window._pending)
    assert bridge.parameters.tune_cons_db == pytest.approx(3.0)
    assert bridge.parameters.tune_mid == pytest.approx(0.8)
    assert bridge.parameters.tune_combined is True
    window.ai_delivery.setCurrentIndex(window.ai_delivery.findData('utterance_lavasr'))
    pump(app, lambda: not window._pending)
    for widget in window._tune_widgets:
        assert not widget.isEnabled()


def test_every_listed_mode_validates():
    experimental = [key for key in DELIVERY_MODES if key not in SHIPPED]
    assert experimental == ['utterance_x_all', 'utterance_x_natural']
    for key in experimental:
        spec = DELIVERY_MODES[key]
        assert spec['delivery'] == 'utterance' and spec['enhancer'] == 'lavasr'
        params = AIParameters(delivery=spec['delivery'], enhancer=spec['enhancer'],
                              experiment=spec['experiment'])
        assert (params.delivery, params.enhancer, params.experiment) == (
            'utterance', 'lavasr', spec['experiment'])


def test_unknown_experiment_is_rejected():
    with pytest.raises(ValueError):
        AIParameters(delivery='utterance', enhancer='lavasr', experiment='nope')


def test_experiment_outside_the_utterance_route_falls_back():
    params = AIParameters(delivery='streaming', enhancer='none', experiment='all')
    assert params.experiment == 'none'
    assert (params.delivery, params.enhancer) == ('streaming', 'none')
    fallback = AIParameters(delivery='streaming', enhancer='none', experiment='natural')
    assert fallback.experiment == 'none'


def test_settings_roundtrip_for_every_experiment():
    for name in EXPERIMENTS:
        values = AppSettings.from_dict({'ai_experiment': name})
        assert values.ai_experiment == name
        assert values.ai_parameters().experiment == name
    assert AppSettings.from_dict({'ai_experiment': 'nope'}).ai_experiment == 'none'


def test_experiment_forces_the_utterance_route_in_settings():
    values = AppSettings.from_dict({'ai_experiment': 'all', 'ai_delivery': 'streaming',
                                    'ai_enhancer': 'none'})
    assert (values.ai_delivery, values.ai_enhancer) == ('utterance', 'lavasr')
    assert values.ai_parameters().experiment == 'all'


class FakeBackend:
    chunk_samples = 2560
    sample_rate = 16000

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def get_stats(self):
        return {'algorithmic_buffer_ms': 960}

    def process_chunk(self, block):
        return np.asarray(block, dtype=np.float32) * 0.5


def speech_with_pause():
    voice = (np.sin(2*np.pi*220*np.arange(8*48000)/48000)*0.3).astype(np.float32)
    pause = np.zeros(2*48000, dtype=np.float32)
    return np.concatenate((voice, pause, voice))


def test_refresh_splits_at_long_pauses_and_preserves_length():
    audio = speech_with_pause()
    segments = split_at_pauses(audio)
    assert len(segments) == 2
    assert segments[0][1] == segments[1][0]
    assert segments[-1][1] == len(audio)
    assert split_at_pauses(np.zeros(48000, dtype=np.float32)) == [(0, 48000)]


def test_refresh_short_utterance_takes_a_single_pass():
    audio = (np.sin(2*np.pi*220*np.arange(8*48000)/48000)*0.3).astype(np.float32)
    backend = FakeBackend()
    stats = {}
    result = convert_utterance(backend, None, None, audio, max_seconds=60,
                               statistics=stats, refresh_pauses=True)
    # Below the length gate the refresh route is the shipped single pass.
    assert stats['refresh_segments'] == 1
    assert len(result) == len(audio)


def test_refresh_conversion_resets_between_segments():
    audio = speech_with_pause()
    backend = FakeBackend()
    stats = {}
    result = convert_utterance(backend, None, None, audio, max_seconds=60,
                               statistics=stats, refresh_pauses=True)
    assert stats['refresh_segments'] == 2
    assert backend.resets >= 4  # open/close per segment, not one pass
    assert len(result) == len(audio) and np.isfinite(result).all()


def test_refresh_without_pauses_is_a_single_pass():
    audio = (np.sin(2*np.pi*220*np.arange(48000)/48000)*0.3).astype(np.float32)
    backend = FakeBackend()
    stats = {}
    result = convert_utterance(backend, None, None, audio, max_seconds=60,
                               statistics=stats, refresh_pauses=True)
    assert stats['refresh_segments'] == 1
    assert len(result) == len(audio)


def test_crossfade_join_keeps_exact_length():
    rng = np.random.default_rng(7)
    parts = [(rng.standard_normal(48000)*0.1).astype(np.float32) for _ in range(3)]
    joined = crossfade_join(parts, 4800)
    assert len(joined) == sum(map(len, parts))-4800*2
    assert np.isfinite(joined).all()
    with pytest.raises(ValueError):
        crossfade_join(parts, 48001)


def test_crossfade_blend_keeps_length():
    tail = np.ones(14400, dtype=np.float32)
    current = np.zeros(48000, dtype=np.float32)
    blended = crossfade_blend(tail, current, 14400)
    assert len(blended) == len(current)
    assert blended[0] == pytest.approx(1.0) and blended[-1] == pytest.approx(0.0)


def test_level_utterance_targets_fixed_rms():
    audio = (np.sin(2*np.pi*220*np.arange(48000)/48000)*0.05).astype(np.float32)
    out = level_utterance(audio)
    rms = float(np.sqrt(np.mean(out.astype(np.float64)**2)))
    assert rms == pytest.approx(10**(-20.0/20), rel=1e-3)
    assert len(out) == len(audio)
    np.testing.assert_array_equal(level_utterance(np.zeros(100, dtype=np.float32)),
                                  np.zeros(100, dtype=np.float32))


def test_blend_highs_preserves_length_and_identity():
    rng = np.random.default_rng(11)
    base = (rng.standard_normal(48000)*0.1).astype(np.float32)
    out = blend_highs(base, base)
    np.testing.assert_allclose(out, base, rtol=1e-4, atol=1e-6)
    other = (rng.standard_normal(48000)*0.1).astype(np.float32)
    mixed = blend_highs(base, other)
    assert len(mixed) == len(base) and np.isfinite(mixed).all()
    assert not np.allclose(mixed, base) and not np.allclose(mixed, other)
    with pytest.raises(ValueError):
        blend_highs(base, other[:1000])


def test_repair_focus_defaults_off_and_validates():
    assert PhraseRepair(0).focus is None
    assert PhraseRepair(0, focus='ending').focus == 'ending'
    with pytest.raises(ValueError):
        PhraseRepair(0, focus='everywhere')


def _sung_pair(seconds=4):
    t = np.arange(seconds*48000)/48000
    vibrato = 1+0.02*np.sin(2*np.pi*5*t)
    source = (np.sin(2*np.pi*180*t*vibrato)*0.3).astype(np.float32)
    voice = (np.sin(2*np.pi*180*t)*0.2).astype(np.float32)
    return source, voice


def test_retrospective_repair_preserves_length_and_bounds_gain():
    source, voice = _sung_pair()
    out, diagnostics = retrospective_repair(source, voice)
    assert len(out) == len(voice) and np.isfinite(out).all()
    assert diagnostics['max_gain_db'] <= 3.0
    assert diagnostics['voiced_frames'] > 0
    with pytest.raises(ValueError):
        retrospective_repair(source, voice[:1000])


def test_retrospective_adaptive_is_identity_on_typical_dynamics():
    source, voice = _sung_pair()
    plain, _ = retrospective_repair(source, voice)
    shaped, diagnostics = retrospective_repair(source, voice, adaptive=True)
    np.testing.assert_array_equal(shaped, plain)
    assert diagnostics['adaptive_caps'] == [2.0, 3.0]


def test_retrospective_adaptive_opens_caps_on_wide_dynamics():
    t = np.arange(4*48000)/48000
    swell = np.where((t % 2.0) < 1.0, 0.6, 0.06)
    source = (np.sin(2*np.pi*180*t)*swell).astype(np.float32)
    voice = (np.sin(2*np.pi*180*t)*0.2).astype(np.float32)
    plain, _ = retrospective_repair(source, voice)
    shaped, diagnostics = retrospective_repair(source, voice, adaptive=True)
    assert diagnostics['adaptive_caps'][0] > 2.0
    assert diagnostics['adaptive_caps'][1] > 3.0
    assert len(shaped) == len(voice) and np.isfinite(shaped).all()
    assert not np.allclose(shaped, plain)
    assert diagnostics['max_gain_db'] <= diagnostics['adaptive_caps'][1]


def test_retrospective_cap_boost_defaults_to_validated_recipe():
    source, voice = _sung_pair()
    base, _ = retrospective_repair(source, voice, adaptive=True)
    same, diagnostics = retrospective_repair(source, voice, adaptive=True, cap_boost=1.0)
    np.testing.assert_array_equal(same, base)
    assert diagnostics['cap_boost'] == 1.0
    t = np.arange(4*48000)/48000
    swell = np.where((t % 2.0) < 1.0, 0.6, 0.06)
    wide = (np.sin(2*np.pi*180*t)*swell).astype(np.float32)
    flat = (np.sin(2*np.pi*180*t)*0.2).astype(np.float32)
    _, low = retrospective_repair(wide, flat, adaptive=True, cap_boost=0.5)
    _, high = retrospective_repair(wide, flat, adaptive=True, cap_boost=1.5)
    assert high['adaptive_caps'][0] > low['adaptive_caps'][0]
    with pytest.raises(ValueError):
        retrospective_repair(source, voice, adaptive=True, cap_boost=2.0)
    with pytest.raises(ValueError):
        retrospective_repair(source, voice, cap_boost=float('nan'))


def test_retrospective_match_rate_defaults_and_bounds():
    source, voice = _sung_pair()
    base, _ = retrospective_repair(source, voice, adaptive=True)
    same, _ = retrospective_repair(source, voice, adaptive=True, match_rate=0.25)
    np.testing.assert_array_equal(same, base)
    untouched, diagnostics = retrospective_repair(source, voice, adaptive=True, match_rate=0.0)
    np.testing.assert_allclose(untouched, voice, rtol=1e-5, atol=1e-7)
    assert diagnostics['matched_frames'] == 0
    with pytest.raises(ValueError):
        retrospective_repair(source, voice, match_rate=0.6)
    with pytest.raises(ValueError):
        retrospective_repair(source, voice, match_rate=float('nan'))


def test_retrospective_pitch_preserves_length_and_bounds():
    source, voice = _sung_pair()
    out, diagnostics = retrospective_pitch(source, voice)
    assert len(out) == len(voice) and np.isfinite(out).all()
    assert diagnostics['max_shift_st'] <= 1.0
    assert diagnostics['voiced_frames'] > 0
    with pytest.raises(ValueError):
        retrospective_pitch(source, voice[:1000])
    with pytest.raises(ValueError):
        retrospective_pitch(source, voice, transfer=1.5)
    with pytest.raises(ValueError):
        retrospective_pitch(source, voice, max_shift_st=3.0)


def test_retrospective_pitch_is_identity_on_matching_contours():
    t = np.arange(4*48000)/48000
    same = (np.sin(2*np.pi*180*t)*0.25).astype(np.float32)
    out, diagnostics = retrospective_pitch(same, same.copy())
    np.testing.assert_allclose(out, same, rtol=1e-4, atol=1e-6)
    assert diagnostics['shifted_grains'] == 0
    off, _ = retrospective_pitch(same, same.copy(), transfer=0.0)
    np.testing.assert_allclose(off, same, rtol=1e-4, atol=1e-6)


def test_retrospective_pitch_follows_source_vibrato():
    t = np.arange(4*48000)/48000
    vibrato = 1+0.04*np.sin(2*np.pi*5*t)
    source = (np.sin(2*np.pi*180*t*vibrato)*0.3).astype(np.float32)
    voice = (np.sin(2*np.pi*180*t)*0.2).astype(np.float32)
    out, diagnostics = retrospective_pitch(source, voice)
    assert len(out) == len(voice) and np.isfinite(out).all()
    assert diagnostics['trusted_frames'] > 0
    assert diagnostics['shifted_grains'] > 0
    assert not np.allclose(out, voice)


def test_utterance_metrics_reports_comparable_numbers():
    source, voice = _sung_pair()
    metrics = utterance_metrics(source, voice)
    assert metrics['voiced_frames'] > 0
    assert -1.0 <= metrics['energy_correlation'] <= 1.0
    assert metrics['voice_f0_range_st'] >= 0.0
    assert 0.0 <= metrics['octave_jump_rate'] <= 1.0


def test_shared_analysis_feeds_repair_and_metrics_once():
    from src.vc.phrase_prosody import analyze_utterance_pair
    source, voice = _sung_pair()
    analysis = analyze_utterance_pair(source, voice)
    out, shaping = retrospective_repair(source, voice, analysis)
    metrics = utterance_metrics(source, voice, analysis)
    assert len(out) == len(voice)
    assert shaping['voiced_frames'] == metrics['voiced_frames']
    with pytest.raises(ValueError):
        analyze_utterance_pair(source, voice[:1000])


def test_suppress_floor_leaves_speech_untouched():
    rng = np.random.default_rng(3)
    speech = (rng.standard_normal(48000)*0.2).astype(np.float32)
    silence = (rng.standard_normal(48000)*0.0005).astype(np.float32)
    audio = np.concatenate((speech, silence))
    out = suppress_floor(audio)
    assert len(out) == len(audio)
    # Speech frames keep their level (away from the smoothed boundary zone);
    # the noise floor drops.
    np.testing.assert_allclose(out[:46000], speech[:46000], rtol=1e-3, atol=1e-6)
    assert float(np.sqrt(np.mean(out[48000:].astype(np.float64)**2))) < \
        float(np.sqrt(np.mean(silence.astype(np.float64)**2)))
    with pytest.raises(ValueError):
        suppress_floor(np.zeros(100, dtype=np.float32))


def test_detect_clicks_counts_only_quiet_jumps():
    quiet = np.zeros(4800, dtype=np.float32)
    click = quiet.copy()
    click[2400] = 0.5
    assert detect_clicks(click) == 1
    assert detect_clicks(quiet) == 0
    loud = (np.sin(2*np.pi*440*np.arange(4800)/48000)*0.5).astype(np.float32)
    assert detect_clicks(loud) == 0


def test_resampler_roundtrip_preserves_voice_band():
    pytest.importorskip('scipy')
    from src.vc.resampler import StreamingResampler
    down, up = StreamingResampler(48000, 16000), StreamingResampler(16000, 48000)
    down.reset(); up.reset()
    # Audit, not a quality bar: lock the current round-trip behaviour so a future
    # filter change shows up here instead of silently moving voice quality.
    for frequency, minimum in ((1000, 0.9), (6000, 0.7)):
        tone = (np.sin(2*np.pi*frequency*np.arange(48000)/48000)*0.5).astype(np.float32)
        # Warm up state the way the live path does, then measure the settled part.
        down.process(tone)
        back = up.process(down.process(tone))
        assert len(back) == len(tone)
        measured = back[4800:9600]
        original = tone[4800:9600]
        ratio = float(np.sqrt(np.mean(measured.astype(np.float64)**2)) /
                      max(float(np.sqrt(np.mean(original.astype(np.float64)**2))), 1e-9))
        assert ratio >= minimum, (frequency, ratio)
    assert np.isfinite(back).all()


def test_integrated_collector_keeps_its_own_segmentation():
    standard = UtteranceCollector()
    assert (standard.silence_frames, standard.keep_frames) == (40, 10)
    seg = UtteranceCollector(silence_ms=500, threshold_db=-42.0, keep_ms=400)
    assert (seg.silence_frames, seg.keep_frames) == (25, 20)
    assert seg.threshold > standard.threshold


def test_integrated_bridge_holds_back_tail_and_crossfades_next_head():
    from src.vc.bridge import AIBridge
    from tests.test_ai_voice import FakeClient, wait
    bridge = AIBridge(AIParameters(delivery='utterance', enhancer='lavasr', experiment='all'),
                      FakeClient)
    overlap = int(0.3*48000)
    voice = np.ones(96000, dtype=np.float32)*0.1
    silence = np.zeros(48000, dtype=np.float32)
    try:
        bridge.load()
        wait(lambda: bridge.status == 'Ready')
        bridge.set_active(True)
        wait(lambda: bridge.ack_generation == bridge.generation)
        for _ in range(2):
            assert bridge.input.write(voice)
            assert bridge.input.write(silence)
            bridge.input_ready.signal()
        wait(lambda: bridge.utterance_stats.get('completed', 0) == 2)
        held = bridge.output.available
        assert held > 0
        # Ending the epoch flushes the held-back tail best-effort.
        bridge.set_active(False)
        wait(lambda: bridge.output.available == held+overlap)
    finally:
        bridge.stop()
    assert not bridge.alive


def test_natural_mode_uses_its_own_collector():
    from src.vc.bridge import AIBridge
    from tests.test_ai_voice import FakeClient, wait
    bridge = AIBridge(AIParameters(delivery='utterance', enhancer='lavasr', experiment='natural'),
                      FakeClient)
    try:
        bridge.load()
        wait(lambda: bridge.status == 'Ready')
    finally:
        bridge.stop()
    assert bridge.parameters.experiment == 'natural'
    assert not bridge.alive


def test_fade_edges_preserves_length_and_fades():
    from src.vc.utterance import fade_edges
    audio = np.ones(48000, dtype=np.float32)
    out = fade_edges(audio)
    assert len(out) == len(audio) and np.isfinite(out).all()
    assert out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(0.0)
    assert out[len(out)//2] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fade_edges(audio[:10], ms=0)


def test_natural_floor_is_gentler_than_all():
    rng = np.random.default_rng(5)
    speech = (rng.standard_normal(48000)*0.2).astype(np.float32)
    silence = (rng.standard_normal(48000)*0.0005).astype(np.float32)
    audio = np.concatenate((speech, silence))
    deep = suppress_floor(audio)
    gentle = suppress_floor(audio, max_cut_db=3.0)
    assert len(gentle) == len(audio) and np.isfinite(gentle).all()
    rms_deep = float(np.sqrt(np.mean(deep[48000:].astype(np.float64)**2)))
    rms_gentle = float(np.sqrt(np.mean(gentle[48000:].astype(np.float64)**2)))
    assert rms_gentle > rms_deep
    np.testing.assert_allclose(gentle[:46000], speech[:46000], rtol=1e-3, atol=1e-6)


def test_natural_finish_preserves_length():
    pytest.importorskip('scipy')
    from src.vc.post_fx import LightPostFX
    from src.vc.utterance import fade_edges
    rng = np.random.default_rng(9)
    audio = (rng.standard_normal(48000)*0.2).astype(np.float32)
    finish = LightPostFX(rate=48000)
    out = finish.process(audio, brightness=50, low_cut=True, limiter=True, enabled=True)
    out = fade_edges(out)
    assert len(out) == len(audio) and np.isfinite(out).all()
    assert float(np.max(np.abs(out))) <= 1.0


def test_service_run_has_no_read_before_local_import():
    """A name read early in run() must not be re-imported later in run().

    Regression test: `from .post_fx import LightPostFX` once sat inside the
    utterance branch while `fx = LightPostFX()` ran earlier, which made the
    name function-local and broke every AI start with "cannot access local
    variable". Pure stdlib: parses the source instead of importing the worker.
    """
    import ast
    from pathlib import Path
    tree = ast.parse(Path('src/vc/service.py').read_text(encoding='utf-8'))
    run = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == 'run')
    # Comprehensions/lambdas are their own scopes: their stores/loads cannot
    # make an UnboundLocalError in run(), so the walk skips those subtrees.
    skip = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda)
    ordered_binds = {arg.arg for arg in run.args.args}
    loads_before_bind = []

    def scan(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, skip):
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    ordered_binds.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                ordered_binds.add(child.name)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                ordered_binds.add(child.id)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in ordered_binds:
                    loads_before_bind.append(child.id)
            scan(child)

    scan(run)
    # Names bound anywhere inside run() that are also loaded before that
    # binding are exactly the UnboundLocalError class. Module-level imports
    # used throughout (e.g. names imported once at top) never appear here.
    all_bound = set()
    for child in ast.walk(run):
        if isinstance(child, skip):
            continue
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            all_bound.update(alias.asname or alias.name.split('.')[0] for alias in child.names)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            all_bound.add(child.name)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            all_bound.add(child.id)
    offenders = sorted({name for name in loads_before_bind if name in all_bound})
    assert offenders == [], f'read before local binding in service.run: {offenders}'


def test_clean_input_removes_rumble_and_dc_but_keeps_voice():
    tone = (np.sin(2*np.pi*440*np.arange(48000)/48000)*0.3).astype(np.float32)
    np.testing.assert_allclose(clean_input(tone), tone, rtol=1e-3, atol=1e-6)
    muddy = (tone+0.2).astype(np.float32)
    out = clean_input(muddy)
    assert abs(float(out.mean())) < 1e-4
    np.testing.assert_allclose(out, tone, rtol=1e-2, atol=1e-4)
    with pytest.raises(ValueError):
        clean_input(tone, cutoff_hz=10.0)
    with pytest.raises(ValueError):
        clean_input(np.zeros(0, dtype=np.float32))


def test_tame_plosives_only_tames_thumps():
    rng = np.random.default_rng(77)
    vowel = (np.sin(2*np.pi*220*np.arange(24000)/48000)*0.3).astype(np.float32)
    bed = (rng.standard_normal(48000)*0.01).astype(np.float32)
    thump = (np.sin(2*np.pi*140*np.arange(4800)/48000)*0.5).astype(np.float32)
    bed[21600:26400] += thump
    audio = np.concatenate((vowel, bed))
    out = tame_plosives(audio)
    assert len(out) == len(audio) and np.isfinite(out).all()
    # The burst body is pressed down; the leading-edge peak is preserved by
    # the fast ramp, so assert energy rather than peak sample.
    region = slice(24000+21600, 24000+26400)
    rms_in = float(np.sqrt(np.mean(audio[region].astype(np.float64)**2)))
    rms_out = float(np.sqrt(np.mean(out[region].astype(np.float64)**2)))
    assert rms_out < rms_in - 0.001
    assert float(np.max(np.abs(out[region]))) <= float(np.max(np.abs(audio[region])))+1e-6
    np.testing.assert_allclose(out[:20000], vowel[:20000], rtol=1e-2, atol=1e-4)
    np.testing.assert_array_equal(tame_plosives(vowel, max_cut_db=0.0), vowel)
    with pytest.raises(ValueError):
        tame_plosives(vowel, max_cut_db=20.0)


def _creaky_tone(seconds=2):
    rng = np.random.default_rng(78)
    t = np.arange(seconds*48000)/48000
    # Fry-like: low rate with slow period wobble plus shimmer, not white FM
    # (which decorrelates into plain noise instead of creak).
    fm = 1+0.08*np.sin(2*np.pi*7*t)
    wave = np.sin(2*np.pi*65*np.cumsum(fm)/48000)
    am = 0.7+0.3*np.sin(2*np.pi*9*t+1.0)
    return ((wave*am*0.25+0.02*rng.standard_normal(len(t)))).astype(np.float32)


def test_fry_fraction_separates_modal_from_creak():
    modal = (np.sin(2*np.pi*180*np.arange(2*48000)/48000)*0.3).astype(np.float32)
    creak = _creaky_tone()
    calm = fry_fraction(modal)
    rough = fry_fraction(creak)
    assert calm['fry'] == 0.0 and calm['frames'] > 0
    assert rough['fry'] > 0.3
    assert rough['periodicity'] < calm['periodicity']
    assert fry_fraction(np.zeros(48000, dtype=np.float32))['fry'] == 0.0
    with pytest.raises(ValueError):
        fry_fraction(np.zeros(0, dtype=np.float32))


def _sibilant_burst(seconds=1):
    rng = np.random.default_rng(21)
    noise = rng.standard_normal(seconds*48000)
    # Crude high-pass via first difference: energy concentrates above 6 kHz.
    burst = np.diff(noise, prepend=0.0)
    burst /= max(float(np.max(np.abs(burst))), 1e-9)
    return (burst*0.3).astype(np.float32)


def test_tame_sibilance_leaves_pure_tones_untouched():
    tone = (np.sin(2*np.pi*440*np.arange(48000)/48000)*0.3).astype(np.float32)
    np.testing.assert_allclose(tame_sibilance(tone), tone, rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(tame_sibilance(tone, max_cut_db=0.0), tone)
    with pytest.raises(ValueError):
        tame_sibilance(np.zeros(0, dtype=np.float32))


def test_tame_sibilance_only_tames_harsh_frames():
    burst = _sibilant_burst()
    out = tame_sibilance(burst)
    assert len(out) == len(burst) and np.isfinite(out).all()
    rms_in = float(np.sqrt(np.mean(burst.astype(np.float64)**2)))
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64)**2)))
    assert rms_out < rms_in
    tone = (np.sin(2*np.pi*440*np.arange(48000)/48000)*0.3).astype(np.float32)
    mixed = np.concatenate((tone, burst))
    shaped = tame_sibilance(mixed)
    np.testing.assert_allclose(shaped[:46000], tone[:46000], rtol=1e-3, atol=1e-6)


def test_lift_consonants_leaves_vowels_and_silence_untouched():
    tone = (np.sin(2*np.pi*220*np.arange(48000)/48000)*0.3).astype(np.float32)
    np.testing.assert_allclose(lift_consonants(tone), tone, rtol=1e-3, atol=1e-6)
    np.testing.assert_array_equal(lift_consonants(tone, max_lift_db=0.0), tone)
    np.testing.assert_array_equal(lift_consonants(np.zeros(48000, dtype=np.float32)),
                                  np.zeros(48000, dtype=np.float32))
    with pytest.raises(ValueError):
        lift_consonants(np.zeros(100, dtype=np.float32), max_lift_db=20.0)
    with pytest.raises(ValueError):
        lift_consonants(np.zeros(48000, dtype=np.float32), band=(100.0, 6000.0))


def test_lift_consonants_lifts_bursts_bounded():
    rng = np.random.default_rng(33)
    vowel = (np.sin(2*np.pi*220*np.arange(24000)/48000)*0.3).astype(np.float32)
    bed = (rng.standard_normal(48000)*0.01).astype(np.float32)
    burst = (rng.standard_normal(7200)*0.3).astype(np.float32)
    bed[20400:27600] += burst
    audio = np.concatenate((vowel, bed))
    out = lift_consonants(audio)
    assert len(out) == len(audio) and np.isfinite(out).all()

    def band_db(x):
        spec = np.abs(np.fft.rfft(x.astype(np.float64)))**2+1e-12
        freqs = np.fft.rfftfreq(len(x), 1.0/48000)
        return 10*np.log10(spec[(freqs >= 2000)&(freqs <= 6000)].sum())

    region = slice(24000+20400, 24000+27600)
    gain = band_db(out[region])-band_db(audio[region])
    assert 0.0 < gain <= 3.5
    np.testing.assert_allclose(out[:20000], vowel[:20000], rtol=1e-2, atol=1e-4)


def test_suppress_floor_protect_exempts_breaths():
    rng = np.random.default_rng(3)
    speech = (rng.standard_normal(48000)*0.2).astype(np.float32)
    silence = (rng.standard_normal(48000)*0.0005).astype(np.float32)
    audio = np.concatenate((speech, silence))
    shield = np.concatenate((np.zeros(48000), np.ones(48000))).astype(np.float64)
    out = suppress_floor(audio, protect=shield)
    assert len(out) == len(audio) and np.isfinite(out).all()
    # Shielded silence keeps its level; unshielded is pressed down.
    np.testing.assert_allclose(out[48000:], silence, rtol=1e-3, atol=1e-6)
    plain = suppress_floor(audio)
    assert float(np.sqrt(np.mean(plain[48000:].astype(np.float64)**2))) < \
        float(np.sqrt(np.mean(out[48000:].astype(np.float64)**2)))
    with pytest.raises(ValueError):
        suppress_floor(audio, protect=np.ones(100))


def test_blend_guard_defaults_reproduce_legacy_mix():
    rng = np.random.default_rng(11)
    base = (rng.standard_normal(48000)*0.1).astype(np.float32)
    other = (rng.standard_normal(48000)*0.1).astype(np.float32)
    np.testing.assert_array_equal(blend_highs(base, other),
                                  blend_highs(base, other, sib_ratio=0.6, sib_mix=0.5))
    np.testing.assert_array_equal(blend_highs(base, other),
                                  blend_highs(base, other, guard_mode='base'))
    strict = blend_highs(base, other, sib_ratio=0.5, sib_mix=0.35)
    assert len(strict) == len(base) and np.isfinite(strict).all()
    with pytest.raises(ValueError):
        blend_highs(base, other, sib_ratio=1.5)
    with pytest.raises(ValueError):
        blend_highs(base, other, guard_mode='bogus')


def test_blend_excess_guard_pulls_back_harsh_restoration():
    t = np.arange(48000)/48000
    base = (np.sin(2*np.pi*440*t)*0.2).astype(np.float32)
    harsh = base.copy()
    harsh[16000:32000] += (np.sin(2*np.pi*8000*t[16000:32000])*0.3).astype(np.float32)
    default = blend_highs(base, harsh)
    excess = blend_highs(base, harsh, guard_mode='excess', sib_mix=0.35)
    assert len(excess) == len(base) and np.isfinite(excess).all()

    def band8(x):
        spec = np.abs(np.fft.rfft(x.astype(np.float64)*np.hanning(len(x))))**2
        freqs = np.fft.rfftfreq(len(x), 1.0/48000)
        return spec[(freqs >= 6000)&(freqs <= 12000)].sum()

    seg = slice(16000, 32000)
    assert band8(excess[seg]) < band8(default[seg])
    # Untouched regions stay identical between the two modes.
    np.testing.assert_allclose(excess[:15000], default[:15000], rtol=1e-4, atol=1e-6)


def test_breath_sample_mask_matches_repair_definition():
    from src.vc.phrase_prosody import analyze_utterance_pair
    t = np.arange(4*48000)/48000
    source = (np.sin(2*np.pi*180*t)*0.3).astype(np.float32)
    voice = (np.sin(2*np.pi*180*t)*0.2).astype(np.float32)
    analysis = analyze_utterance_pair(source, voice)
    mask = breath_sample_mask(analysis, len(source))
    assert mask.shape == source.shape and mask.dtype == np.float32
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
    with pytest.raises(ValueError):
        breath_sample_mask(analysis, 0)
    with pytest.raises(ValueError):
        breath_sample_mask(dict(times=[], voiced=[], source_features=[],
                                voice_features=[]), len(source))
