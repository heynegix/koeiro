import shutil
import wave

import numpy as np
import pytest

from src.vc import voice_quality as vq

RATE = 16000


def speech(seconds, f0=180.0, level=0.08, rate=RATE, vibrato_hz=4.7, depth=0.04):
    """A harmonic stack with true vibrato, so f0 has a known ground truth.

    The phase is integrated from the instantaneous frequency; multiplying ``t`` by a
    vibrato factor instead would be phase modulation whose frequency sweeps upward.
    """
    n = int(seconds*rate)
    t = np.arange(n)/rate
    phase = 2*np.pi*f0*(t + depth*(1-np.cos(2*np.pi*vibrato_hz*t))/(2*np.pi*vibrato_hz))
    out = np.zeros(n)
    for harmonic, amplitude in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.2), (6, 0.12), (9, 0.07)):
        out += amplitude*np.sin(harmonic*phase)
    return (level*out).astype(np.float32)


def write_wav(path, audio, rate=RATE):
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.clip(audio, -1, 1)*32767).astype('<i2').tobytes())
    return path


# --- decode ---

@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg is required for decoding')
def test_decode_accepts_any_container_ffmpeg_reads_not_just_known_suffixes(tmp_path):
    # A WAV behind a meaningless extension must still register; decodability decides.
    source = write_wav(tmp_path/'voice.dat', speech(4.0))
    audio, scale = vq.decode_audio(source)
    assert audio.dtype == np.float32
    assert audio.shape == (4*RATE,)
    assert scale == 1.0
    assert np.isfinite(audio).all()


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg is required for decoding')
def test_decode_rescales_inter_sample_overs_and_bounds_length(tmp_path):
    audio, scale = vq.decode_audio(write_wav(tmp_path/'loud.wav', speech(4.0)*20))
    assert 0 < scale < 1
    assert float(np.max(np.abs(audio))) <= vq.SAFE_PEAK + 1e-6
    with pytest.raises(ValueError, match='長すぎます'):
        vq.decode_audio(write_wav(tmp_path/'ok.wav', speech(4.0)), max_seconds=1.0)
    with pytest.raises(ValueError):
        vq.decode_audio(tmp_path/'missing.wav')


# --- speech detection ---

def test_robust_floor_finds_speech_the_legacy_percentile_floor_misses():
    # Near-continuous speech: the 20th percentile is a speech frame, so the legacy
    # floor rises with the signal and the threshold ends up above the speech.
    audio = np.concatenate([speech(30.0, level=0.05), np.zeros(int(0.8*RATE), np.float32)])
    assert vq.detect_speech(audio, RATE, robust=False) == []
    regions = vq.detect_speech(audio, RATE, robust=True)
    assert regions
    covered = sum(end-start for start, end in regions)/RATE
    assert covered > 25.0


def test_candidate_regions_cut_at_clipping_boundaries():
    audio = np.concatenate([np.zeros(2*RATE), speech(5.0), np.ones(3*RATE)])
    regions = vq.candidate_regions(audio, RATE)
    assert regions
    # A boundary is opened exactly where the full-scale tail starts, so the clean speech
    # ends before it. The clipped run itself stays a candidate: rejecting it with a
    # reason is the gate's job and keeps the tally in the report honest.
    boundaries = {start for start, _ in regions} | {end for _, end in regions}
    assert 7.0*RATE in boundaries
    assert regions[0][1] == 7.0*RATE
    # Nothing that survives the gates may carry clipping.
    parts, _, _ = vq.select_reference(audio, RATE)
    assert parts
    assert all(float(np.max(np.abs(part))) < 0.999 for part in parts)


def test_split_region_never_exceeds_the_maximum_length():
    audio = np.concatenate([np.zeros(int(0.5*RATE)), speech(40.0), np.zeros(int(0.5*RATE))])
    for start, end in vq.candidate_regions(audio, RATE):
        assert (end-start)/RATE <= 10.05


# --- quality gates ---

@pytest.mark.parametrize('audio,expected', [
    (np.zeros(6*RATE, np.float32), 'near_silent'),
    (speech(6.0)*0.0056, 'near_silent'),
    (np.clip(speech(6.0)*20, -1, 1), 'clipped'),
])
def test_gates_reject_unusable_material(audio, expected):
    assert vq.rejection_reason(vq.analyse(audio, RATE)) == expected


def test_gates_reject_continuous_hiss_through_flatness():
    hiss = np.random.default_rng(0).standard_normal(int(6*RATE)).astype(np.float32)*0.05
    stats = vq.analyse(hiss, RATE)
    assert stats['spectral_flatness'] > vq.MAX_SPECTRAL_FLATNESS
    assert vq.rejection_reason(stats) is not None
    # Speech on a continuous noise bed has no measurable SNR, so flatness is what
    # carries it. The gate sits at 0.25, anchored on real speech (measured maximum
    # 0.165). Against this more tonal synthetic voice the transition is sharp: 0.05 of
    # noise measures 0.206 and stays acceptable, 0.07 measures 0.300 and is rejected.
    buried = speech(6.0) + np.random.default_rng(1).standard_normal(int(6*RATE)).astype(np.float32)*0.07
    buried_stats = vq.analyse(buried, RATE)
    assert buried_stats['snr_db'] is None
    assert buried_stats['voiced_ratio'] >= 0.10
    assert vq.rejection_reason(buried_stats) == 'hiss_or_noise'
    assert vq.rejection_reason(vq.analyse(
        speech(6.0)+np.random.default_rng(1).standard_normal(int(6*RATE)).astype(np.float32)*0.05, RATE)) is None


def test_gate_separates_speech_from_masked_speech_without_rejecting_clean_speech():
    clean = vq.analyse(speech(6.0), RATE)
    assert vq.rejection_reason(clean) is None
    assert clean['spectral_flatness'] < vq.MAX_SPECTRAL_FLATNESS
    mild = vq.analyse(speech(6.0)+np.random.default_rng(2).standard_normal(6*RATE).astype(np.float32)*0.004, RATE)
    assert vq.rejection_reason(mild) is None


def test_quality_score_is_bounded_and_tolerates_unmeasurable_snr():
    stats = vq.analyse(speech(6.0), RATE)
    stats['snr_db'] = None
    assert 0.0 <= vq.quality_score(stats) <= 1.0
    stats['snr_db'] = 500.0
    assert 0.0 <= vq.quality_score(stats) <= 1.0


@pytest.mark.parametrize('f0', [100, 120, 150, 190, 240, 300, 400])
def test_f0_estimate_is_accurate_against_a_known_signal(f0):
    values, voiced = vq.estimate_f0(speech(4.0, f0=f0), RATE)
    voiced_values = values[values > 0]
    assert voiced_values.size
    assert voiced > 0.9
    assert float(np.median(voiced_values)) == pytest.approx(f0, rel=0.03)


def test_analyse_rejects_non_finite_and_multichannel():
    with pytest.raises(ValueError):
        vq.analyse(np.full(4*RATE, np.nan), RATE)
    with pytest.raises(ValueError):
        vq.analyse(np.zeros((4*RATE, 2)), RATE)


# --- selection ---

def test_select_reference_respects_budget_and_keeps_chronological_order():
    audio = np.concatenate([speech(12.0), np.zeros(int(1.5*RATE), np.float32), speech(12.0, f0=230.0)])
    parts, selections, report = vq.select_reference(audio, RATE, budget_seconds=12.0)
    assert report['retained_seconds'] <= 12.0 + 1e-6
    assert selections == sorted(selections, key=lambda row: row['offset_seconds'])
    assert sum(len(part) for part in parts)/RATE == pytest.approx(report['retained_seconds'], abs=0.05)
    assert all(2.0 <= row['duration_seconds'] <= 10.05 for row in selections)
    assert all(vq.rejection_reason(vq.analyse(part, RATE)) is None for part in parts)


def test_select_reference_reports_why_nothing_qualified():
    hiss = np.random.default_rng(3).standard_normal(int(20*RATE)).astype(np.float32)*0.05
    with pytest.raises(vq.NoUsableAudio) as failure:
        vq.select_reference(hiss, RATE)
    report = failure.value.report
    assert report['accepted_count'] == 0 and report['rejected']
    assert report['advice'] and report['selections'] == []


def test_select_reference_needs_a_few_seconds_of_audio():
    with pytest.raises(ValueError, match='短すぎます'):
        vq.select_reference(speech(2.0), RATE)


def test_diversity_prefers_clips_that_are_not_already_covered():
    # Three very different voices separated by pauses longer than the merge gap, so
    # each stays its own region. The selection should span all three rather than
    # taking three clips of the loudest one.
    gap = np.zeros(int(0.8*RATE), np.float32)
    audio = np.concatenate([gap, speech(6.0, f0=120.0), gap,
                            speech(6.0, f0=190.0), gap, speech(6.0, f0=300.0)])
    _, selections, _ = vq.select_reference(audio, RATE, budget_seconds=20.0)
    assert len(selections) == 3
    assert [row['f0_median_hz'] for row in selections] == pytest.approx([120, 190, 300], abs=3)
    assert max(row['diversity_to_previous'] for row in selections[1:]) > 0.1


def test_short_pauses_are_treated_as_one_utterance():
    # A pause shorter than the merge gap belongs to the same utterance, so it must not
    # be split into separate clips. The utterance stays under the 10s maximum length.
    gap = np.zeros(int(0.2*RATE), np.float32)
    audio = np.concatenate([speech(4.0, f0=190.0), gap, speech(4.0, f0=190.0)])
    _, selections, _ = vq.select_reference(audio, RATE, budget_seconds=60.0)
    assert len(selections) == 1
    assert selections[0]['duration_seconds'] > 7.0


def test_profile_distance_is_zero_for_identical_material_and_large_for_unrelated():
    first = vq.acoustic_profile(speech(4.0, f0=150.0), RATE)
    same = vq.acoustic_profile(speech(4.0, f0=150.0), RATE)
    other = vq.acoustic_profile(speech(4.0, f0=320.0), RATE)
    assert vq.profile_distance(first, same) == pytest.approx(0.0, abs=1e-9)
    assert vq.profile_distance(first, other) > 0.05


def test_report_is_json_serialisable():
    import json
    audio = np.concatenate([speech(8.0), np.zeros(int(1.0*RATE), np.float32), speech(8.0)])
    _, _, report = vq.select_reference(audio, RATE)
    assert json.loads(json.dumps(report, ensure_ascii=False)) == report


def test_registration_report_is_rendered_for_the_dialog():
    from src.gui.voice_registration import format_report
    text = format_report({'retained_seconds': 48.41, 'budget_seconds': 60.0, 'source_seconds': 59.74,
                          'speech_seconds': 48.41, 'region_count': 12, 'candidate_count': 12,
                          'accepted_count': 12, 'rejected': {'clipped': 2}, 'advice': ['テスト'],
                          'selections': [{'offset_seconds': 0.52, 'duration_seconds': 3.0,
                                          'quality': 0.58, 'snr_db': 17.0, 'f0_median_hz': 176.4},
                                         {'offset_seconds': 5.09, 'duration_seconds': 2.03,
                                          'quality': 0.77, 'snr_db': None, 'f0_median_hz': None}]})
    assert '48.41秒' in text and '上限 60秒' in text
    assert 'clipped×2' in text
    assert '17dB' in text and '176Hz' in text
    assert '※ テスト' in text
    assert format_report(None) == '' and format_report({}) == ''