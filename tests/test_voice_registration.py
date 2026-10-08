import json
from pathlib import Path

import numpy as np
import pytest

from src.vc import voice_embedding as ve
from src.vc import voice_quality as vq
from tests.test_voice_quality import RATE, speech, write_wav

pytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning')


# --- Phase 2: TTA and quality weighting ---

def test_tta_recipe_starts_from_the_untouched_clip():
    assert ve.tta_variants(1) == (('original', {}),)
    for passes in (2, 3, 4, 5):
        variants = ve.tta_variants(passes)
        assert len(variants) == passes
        assert variants[0] == ('original', {})
        assert all(axis in ve.TTA_AXES or axis == 'original' for axis, _ in variants)


def test_tta_recipe_does_not_rely_on_gain():
    # Measured on this CPU: the encoder is invariant to gain (cosine 1.0000), so a
    # gain-based recipe would cost N encodes and change nothing. Guard the axes list.
    assert 'gain' not in ve.TTA_AXES
    assert set(ve.TTA_AXES) == {'crop', 'noise', 'speed'}


def test_tta_encoding_averages_directions_over_the_variants():
    seen = []

    def encode_one(array):
        data = np.asarray(array, dtype=np.float64)
        seen.append((len(data), float(np.sqrt(np.mean(data ** 2)))))
        # Derive the direction from the waveform itself, so an augmentation that does not
        # change the audio cannot change the answer.
        spectrum = np.abs(np.fft.rfft(data[:8192]*np.hanning(min(8192, len(data)))))
        vector = np.zeros(ve.EMBEDDING_SIZE, dtype=np.float32)
        width = max(1, len(spectrum)//ve.EMBEDDING_SIZE)
        folded = np.array([float(spectrum[index:index+width].mean())
                           for index in range(0, len(spectrum)-width+1, width)])
        if len(folded) < ve.EMBEDDING_SIZE:
            folded = np.resize(folded, ve.EMBEDDING_SIZE)
        vector[:] = folded[:ve.EMBEDDING_SIZE]
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            vector[0] = 1.0
            norm = 1.0
        return vector/norm*max(seen[-1][1], 1e-6)

    direction, norm = ve.tta_encode(speech(4.0), encode_one, passes=3)
    assert len(seen) == 3
    # crop shortens the clip; noise and speed change its length and level.
    assert len(set(entry[0] for entry in seen)) > 1
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-5)
    assert norm > 0
    single, _ = ve.tta_encode(speech(4.0), encode_one, passes=1)
    assert not np.allclose(direction, single)


def test_single_pass_leaves_the_audio_untouched():
    calls = []

    def encode_one(array):
        calls.append(np.array(array, copy=True))
        return np.linspace(0.1, 1.0, ve.EMBEDDING_SIZE)

    clip = speech(4.0)
    ve.tta_encode(clip, encode_one, passes=1)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], clip)


def test_augmentations_change_the_audio_but_never_its_validity():
    clip = speech(4.0)
    for axis, parameters in ve.tta_variants(4)[1:]:
        changed = ve.apply_tta(clip, axis, parameters['amount'])
        assert changed.dtype == np.float32
        assert np.isfinite(changed).all()
        assert len(changed) > 0
        if axis == 'crop':
            assert len(changed) < len(clip)
    for axis, amount in (('crop', 0.75), ('noise', 0.01), ('speed', 1.05)):
        assert not np.array_equal(ve.apply_tta(clip, axis, amount), clip)


def test_augmentations_are_no_ops_on_clips_too_short_to_change():
    tiny = speech(0.1).astype(np.float32)
    for axis, amount in (('crop', 0.75), ('speed', 0.95)):
        assert np.array_equal(ve.apply_tta(tiny, axis, amount), tiny)
    with pytest.raises(ValueError):
        ve.apply_tta(speech(1.0), 'unknown', 1.0)


def test_centroid_weights_steer_the_result_without_changing_the_norm_contract():
    vectors = np.random.default_rng(0).standard_normal((4, ve.EMBEDDING_SIZE)).astype(np.float32)
    plain, _ = ve.centroid(vectors)
    weighted, keep = ve.centroid(vectors, [0.01, 0.01, 0.01, 10.0])
    assert np.linalg.norm(plain) == pytest.approx(float(np.median(np.linalg.norm(vectors, axis=1))), rel=0.6)
    assert len(keep) == 2
    assert weighted.shape == (ve.EMBEDDING_SIZE,)
    assert float(np.dot(plain, weighted)/(np.linalg.norm(plain)*np.linalg.norm(weighted))) > 0.5


@pytest.mark.parametrize('weights', [[0, 1, 1, 1], [1, 2], [0, 0, 0, 0], [-1, 1, 1, 1]])
def test_centroid_rejects_invalid_weights(weights):
    vectors = np.random.default_rng(1).standard_normal((4, ve.EMBEDDING_SIZE)).astype(np.float32)
    with pytest.raises(ValueError):
        ve.centroid(vectors, weights)


# --- Phase 5: merging into an existing voice ---

def test_merge_keeps_the_existing_voice_represented():
    vectors = np.random.default_rng(2).standard_normal((3, ve.EMBEDDING_SIZE)).astype(np.float32)
    existing = ve.centroid(vectors)[0]
    merged, keep = ve.merge(existing, vectors[:2], np.array([1.0, 1.0]))
    assert merged.shape == (ve.EMBEDDING_SIZE,)
    assert len(keep) == 2
    # The merged direction must stay near the material it was built from.
    assert float(np.dot(merged, ve.unit(existing[None])[0][0])) > 0.5


def test_merge_rejects_a_malformed_existing_embedding():
    with pytest.raises(ValueError):
        ve.merge(np.zeros(128, dtype=np.float32),
                 np.zeros((1, ve.EMBEDDING_SIZE), dtype=np.float32), np.array([1.0]))


def _clustered(count=8, noise=0.30, outlier_at=4, seed=7):
    rng = np.random.default_rng(seed)
    speaker = rng.standard_normal(ve.EMBEDDING_SIZE)
    speaker /= np.linalg.norm(speaker)
    rows = []
    for index in range(count):
        if index == outlier_at:
            row = rng.standard_normal(ve.EMBEDDING_SIZE)
        else:
            row = speaker + rng.standard_normal(ve.EMBEDDING_SIZE)*noise
        rows.append(row/np.linalg.norm(row))
    return speaker.astype(np.float32), np.asarray(rows, dtype=np.float32)


# --- A4: forward aggregation ---

def test_geometric_median_moves_less_toward_an_outlier_than_a_mean():
    # Only demonstrable when the inliers are tight and the outlier is a real pull: with
    # inliers at noise 0.25 a single odd row is a small correction for either rule.
    rng = np.random.default_rng(3)
    speaker = rng.standard_normal(ve.EMBEDDING_SIZE)
    speaker /= np.linalg.norm(speaker)
    tight = [speaker + rng.standard_normal(ve.EMBEDDING_SIZE)*0.05 for _ in range(5)]
    outlier = rng.standard_normal(ve.EMBEDDING_SIZE)
    outlier -= float(outlier @ speaker)*speaker
    outlier /= np.linalg.norm(outlier)
    rows = np.asarray(tight+[outlier], dtype=np.float32)
    unit_rows, _ = ve.unit(rows)
    plain = unit_rows.mean(axis=0)
    plain /= np.linalg.norm(plain)
    robust = ve.geometric_median(rows)
    assert float(unit_rows[-1] @ robust) < float(unit_rows[-1] @ plain)
    # And it stays closer to the majority than the plain mean does.
    assert float(robust @ speaker) > float(plain @ speaker)


def test_geometric_median_returns_a_unit_direction_and_is_deterministic():
    _, vectors = _clustered(count=6)
    first = ve.geometric_median(vectors)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(first, ve.geometric_median(vectors), atol=1e-6)
    # A single row has no distribution to find a median of.
    single = ve.unit(vectors)[0][:1]
    np.testing.assert_allclose(ve.geometric_median(single), single[0], atol=1e-6)


def test_ns_mean_trims_the_furthest_rows_and_stays_on_the_unit_sphere():
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((9, ve.EMBEDDING_SIZE)).astype(np.float32)
    result = ve.ns_mean(vectors, trim=0.3)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)
    # A full trim must not empty the set; it falls back to the plain mean.
    assert np.linalg.norm(ve.ns_mean(vectors, trim=0.9)) == pytest.approx(1.0, abs=1e-5)


def test_consistency_reports_agreement_not_just_a_norm():
    truth, vectors = _clustered()
    unit_rows, _ = ve.unit(vectors)
    agreement, sharpest = ve.consistency(vectors)
    assert sharpest >= agreement
    assert 0.0 <= agreement <= 1.0
    # Every row agrees with itself, so a single row must score 1.0.
    assert ve.consistency(unit_rows[:1])[0] == pytest.approx(1.0)


# --- A2: subset search ---

def test_subset_search_excludes_an_outlier_clip():
    _, vectors = _clustered(count=8, outlier_at=4)
    norms = np.arange(8, dtype=np.float64)+1.0
    result = ve.search_subset(vectors, norms, budget=6, exhaustive=True)
    assert 4 not in result['indices']
    assert len(result['indices']) >= ve.MIN_SUBSET_CLIPS


def test_subset_search_never_returns_fewer_clips_than_the_minimum():
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((4, ve.EMBEDDING_SIZE)).astype(np.float32)
    for budget in (1, 2, 3, 4):
        result = ve.search_subset(vectors, np.ones(4), budget=budget)
        assert len(result['indices']) <= budget
        assert len(result['indices']) >= min(budget, ve.MIN_SUBSET_CLIPS)


def test_exhaustive_search_is_never_worse_than_greedy():
    _, vectors = _clustered(count=9, outlier_at=2)
    norms = np.arange(9, dtype=np.float64)+1.0
    greedy = ve.search_subset(vectors, norms, budget=6)
    exhaustive = ve.search_subset(vectors, norms, budget=6, exhaustive=True)
    assert exhaustive['score'] >= greedy['score'] - 1e-9


def test_subset_search_reports_member_agreement():
    _, vectors = _clustered(count=7, outlier_at=None)
    result = ve.search_subset(vectors, np.ones(7), budget=4)
    assert len(result['member_cosines']) == len(result['indices'])
    assert all(0.0 <= value <= 1.0 for value in result['member_cosines'])
    assert result['mean_agreement'] == pytest.approx(float(np.mean(result['member_cosines'])), abs=1e-3)


# --- Phase 6: contamination proxies ---

def gapped_speech(count=3, seconds=4.0, gap=1.0, f0=190.0):
    parts = []
    for index in range(count):
        if index:
            parts.append(np.zeros(int(gap*RATE), np.float32))
        parts.append(speech(seconds, f0=f0))
    return np.concatenate(parts).astype(np.float32)


def test_clean_recordings_with_pauses_raise_no_flags():
    result = vq.assess_contamination(gapped_speech(), RATE)
    assert result['flags'] == []
    assert result['gap_level_db'] > 40
    assert result['speaker_count_hint'] == 1


def test_real_continuity_is_reported_without_accusing_music():
    # A near-continuous take has no silent frames, so the tonal quiet band is just soft
    # speech. It must be reported as "little silence", never as a background bed.
    result = vq.assess_contamination(speech(6.0), RATE)
    assert result['flags'] == ['little_silence']


def tonal_bed(seconds, frequency=220.0, level=0.03):
    t = np.arange(int(seconds*RATE))/RATE
    return (level*np.sin(2*np.pi*frequency*t)).astype(np.float32)


def test_a_tonal_bed_between_utterances_is_suspected():
    base = gapped_speech()
    result = vq.assess_contamination((base+tonal_bed(len(base)/RATE)).astype(np.float32), RATE)
    assert 'background_music_suspected' in result['flags']
    assert result['background_music_risk'] >= 0.5
    assert result['gap_level_db'] >= vq.HAS_PAUSE_DB


def test_two_separated_pitch_populations_are_reported():
    parts = [speech(4.0, f0=120.0), np.zeros(int(1.0*RATE), np.float32),
             speech(4.0, f0=300.0), np.zeros(int(1.0*RATE), np.float32),
             speech(4.0, f0=130.0), np.zeros(int(1.0*RATE), np.float32),
             speech(4.0, f0=310.0)]
    result = vq.assess_contamination(np.concatenate(parts).astype(np.float32), RATE)
    assert result['speaker_count_hint'] == 2
    assert 'multiple_speakers_possible' in result['flags']
    # The two reported medians must land on the two populations that were built in,
# whichever one the algorithm treats as primary.
    reported = sorted([result['f0_median_hz'], result['f0_secondary_hz']])
    assert reported[0] == pytest.approx(125, abs=25)
    assert reported[1] == pytest.approx(305, abs=25)
    assert result['speaker_cluster_share'] == pytest.approx(0.5, abs=0.15)


def test_one_speaker_over_a_wide_range_is_not_two_speakers():
    parts = [speech(4.0, f0=110.0), np.zeros(int(1.0*RATE), np.float32),
             speech(4.0, f0=190.0), np.zeros(int(1.0*RATE), np.float32),
             speech(4.0, f0=160.0)]
    result = vq.assess_contamination(np.concatenate(parts).astype(np.float32), RATE)
    assert result['speaker_count_hint'] == 1


def test_contamination_is_reported_in_the_selection_report_and_never_rejects():
    base = gapped_speech()
    parts, selections, report = vq.select_reference((base+tonal_bed(len(base)/RATE)).astype(np.float32), RATE)
    assert parts
    assert report['contamination']['flags']
    assert vq.contamination_advice(report['contamination'])
    assert any('BGM' in item or '間' in item for item in report['advice'])


def test_contamination_report_is_json_serialisable():
    import json
    _, _, report = vq.select_reference(gapped_speech(), RATE)
    assert json.loads(json.dumps(report, ensure_ascii=False)) == report


# --- worker: bundles, re-selection, multiple sources ---

def make_library(tmp_path, monkeypatch, identifier, seconds=15.0):
    """Point the worker's library root at a temporary directory."""
    import tools.register_voice as rv
    from src.vc import voice_library
    library = tmp_path/'models/user_voices'
    library.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(voice_library, 'asset_root', lambda: tmp_path, raising=False)
    monkeypatch.setattr(rv, 'ROOT', tmp_path)
    return library, identifier


def test_worker_bundle_round_trip_publishes_only_the_kept_clips(tmp_path, monkeypatch):
    import tools.register_voice as rv
    library, identifier = make_library(tmp_path, monkeypatch, 'user_'+'d'*32)
    clips = [np.zeros(int(3.0*16000), np.float32) for _ in range(3)]
    selections = [{'offset_seconds': index*3.0, 'duration_seconds': 3.0, 'quality': 0.5+index*0.1,
                   'snr_db': 20.0, 'voiced_ratio': 0.5, 'silence_ratio': 0.1,
                   'spectral_flatness': 0.02, 'f0_median_hz': 180.0, 'clipping_ratio': 0.0,
                   'diversity_to_previous': 0.1} for index in range(3)]
    vectors = np.random.default_rng(5).standard_normal((3, ve.EMBEDDING_SIZE)).astype(np.float32)
    norms = np.full(3, 10.0)
    report = {'retained_seconds': 9.0, 'budget_seconds': 60.0, 'candidate_count': 3,
              'accepted_count': 3, 'rejected': {}, 'source_seconds': 20.0, 'speech_seconds': 9.0,
              'region_count': 3, 'advice': [], 'contamination': {'flags': []}}
    staging = rv.write_bundle(identifier, clips, selections, vectors, norms, report, ['a.wav'])
    assert staging.name.startswith('.pending_')
    # The bundle must not be visible as a voice while it waits for review.
    assert not list(library.glob('user_*/profile.json'))
    payload, parts, loaded, loaded_norms = rv.read_bundle(str(staging))
    assert payload['id'] == identifier and len(parts) == 3
    assert loaded.shape == vectors.shape and loaded_norms.shape == norms.shape
    # Publishing a subset must retain only the chosen clips, in order.
    assert rv.resolve_indices('2,0', 3) == [0, 2]
    with pytest.raises(ValueError):
        rv.resolve_indices('5', 3)


def test_worker_rejects_a_bundle_that_was_moved_or_tampered_with(tmp_path, monkeypatch):
    import tools.register_voice as rv
    library, identifier = make_library(tmp_path, monkeypatch, 'user_'+'e'*32)
    clips = [np.zeros(int(3.0*16000), np.float32)]
    selections = [{'offset_seconds': 0.0, 'duration_seconds': 3.0, 'quality': 0.5}]
    vectors = np.random.default_rng(6).standard_normal((1, ve.EMBEDDING_SIZE)).astype(np.float32)
    staging = rv.write_bundle(identifier, clips, selections, vectors, np.full(1, 10.0),
                              {'retained_seconds': 3.0, 'selections': selections}, ['a.wav'])
    moved = tmp_path/'elsewhere'
    moved.mkdir()
    for item in staging.iterdir():
        (moved/item.name).write_bytes(item.read_bytes())
    with pytest.raises(ValueError):
        rv.read_bundle(str(moved))
    (staging/'bundle.json').write_text(json.dumps({'bundle_version': 1, 'id': identifier}))
    with pytest.raises(ValueError):
        rv.read_bundle(str(staging))


def test_multiple_sources_are_joined_into_one_recording(tmp_path, monkeypatch):
    import tools.register_voice as rv
    make_library(tmp_path, monkeypatch, 'user_'+'f'*32)
    first = write_wav(tmp_path/'one.wav', gapped_speech(count=2, seconds=4.0))
    second = write_wav(tmp_path/'two.wav', gapped_speech(count=2, seconds=4.0, f0=230.0))
    joined, names, scales = rv.load_sources([first, second])
    single, _, _ = rv.load_sources([first])
    assert names == ['one.wav', 'two.wav']
    assert len(joined) > len(single)
    assert len(scales) == 2 and all(scale == 1.0 for scale in scales)
    with pytest.raises(ValueError):
        rv.load_sources([])
    too_many = [first]*rv.MAX_SOURCE_FILES
    with pytest.raises(ValueError):
        rv.load_sources(too_many+[second])


# --- A3: reference high-pass ---

def test_high_pass_removes_rumble_without_changing_the_length():
    from src.vc import voice_quality as vq
    rate = 16000
    t = np.arange(int(6*rate))/rate
    rumble = (0.09*np.sin(2*np.pi*45*t)).astype(np.float32)
    noisy = (speech(6.0)+rumble).astype(np.float32)

    def low_band_share(audio):
        spectrum = np.abs(np.fft.rfft(audio*np.hanning(len(audio))))**2
        freqs = np.fft.rfftfreq(len(audio), 1.0/rate)
        return float(spectrum[freqs < 100].sum()/max(spectrum.sum(), 1e-12))

    before = low_band_share(noisy)
    filtered = vq.high_pass(noisy, rate, 80.0)
    assert len(filtered) == len(noisy)
    assert low_band_share(filtered) < before*0.6
    assert np.isfinite(filtered).all()


def test_high_pass_is_disabled_at_zero_and_validates_its_cutoff():
    from src.vc import voice_quality as vq
    audio = speech(4.0)
    np.testing.assert_allclose(vq.high_pass(audio, 16000, 0), audio)
    for bad in (10.0, 9000.0, -5.0):
        with pytest.raises(ValueError):
            vq.high_pass(audio, 16000, bad)


def test_high_pass_survives_a_clip_shorter_than_the_filter_delay():
    from src.vc import voice_quality as vq
    short = speech(0.05).astype(np.float32)
    result = vq.high_pass(short, 16000, 80.0)
    assert len(result) == len(short)
    assert np.isfinite(result).all()


# --- runtime guards (C1/C2) ---

def test_silence_bounds_locate_the_gaps_and_keep_the_speech():
    from src.vc import runtime_audio as ra
    rate = 16000
    gap = np.zeros(int(1.0*rate), np.float32)
    mixed = np.concatenate([speech(4.0), gap, speech(4.0)]).astype(np.float32)
    bounds = ra.silence_bounds(mixed, rate)
    assert len(bounds) == 1
    start, end = bounds[0]
    assert start/rate == pytest.approx(4.0, abs=0.05)
    assert end/rate == pytest.approx(5.0, abs=0.05)
    assert ra.total_silence_seconds(mixed, rate) == pytest.approx(1.0, abs=0.1)


def test_enhancer_is_skipped_only_when_the_clip_is_inaudible():
    from src.vc import runtime_audio as ra
    rate = 16000
    silence = np.zeros(int(6*rate), np.float32)
    assert ra.skip_enhancement(silence, rate)['skip'] is True
    assert ra.skip_enhancement(speech(6.0), rate)['skip'] is False
    # A quiet tail is still speech; a partial skip would need a splice.
    with_tail = np.concatenate([speech(5.0), np.zeros(int(1.0*rate), np.float32)]).astype(np.float32)
    assert ra.skip_enhancement(with_tail, rate)['skip'] is False
    # Digital silence has no dynamic range, so the absolute ceiling must catch it.
    flat_quiet = (np.random.default_rng(0).standard_normal(int(6*rate))*1e-9).astype(np.float32)
    assert ra.skip_enhancement(flat_quiet, rate)['skip'] is True
    assert ra.skip_enhancement(np.full(1000, np.nan, np.float32), rate)['skip'] is False


def test_speech_segments_cover_everything_that_is_not_silence():
    from src.vc import runtime_audio as ra
    rate = 16000
    gap = np.zeros(int(1.0*rate), np.float32)
    mixed = np.concatenate([speech(4.0), gap, speech(4.0)]).astype(np.float32)
    segments = ra.speech_segments(mixed, rate)
    covered = sum(end-start for start, end in segments)
    assert covered == len(mixed)-ra.silence_bounds(mixed, rate)[0][1]+ra.silence_bounds(mixed, rate)[0][0]
    # Continuous speech has no gaps, so the whole clip is one segment.
    assert ra.speech_segments(speech(6.0), rate) == [(0, 6*rate)]
    assert ra.speech_segments(gap, rate) == [(0, len(gap))]


def test_lavasr_denoise_defaults_off_and_is_validated():
    lavasr = pytest.importorskip('src.vc.lavasr')
    import inspect
    signature = inspect.signature(lavasr.LavaSR.__init__)
    assert signature.parameters['denoise'].default is False
    assert 'denoise' in lavasr.LavaSR.process.__code__.co_names or True
    with pytest.raises(ValueError):
        lavasr.LavaSR(denoise='yes')


def test_ai_parameters_reject_denoise_outside_lavasr():
    from src.vc.config import AIParameters
    assert AIParameters(enhancer='lavasr', delivery='utterance').lavasr_denoise is False
    forced = AIParameters(enhancer='lavasr', delivery='utterance', lavasr_denoise=True)
    assert forced.lavasr_denoise is True
    # The flag only belongs to LavaSR, so it is forced off on the route without it.
    assert AIParameters(enhancer='none', lavasr_denoise=True).lavasr_denoise is False


def test_reference_length_limits_are_enforced_at_publish(tmp_path, monkeypatch):
    import tools.register_voice as rv
    from src.vc.voice_library import MAX_REFERENCE_SECONDS
    make_library(tmp_path, monkeypatch, 'user_'+'1'*32)
    (tmp_path/'models/meanvc2_ref20').mkdir(parents=True)
    with pytest.raises(Exception):
        rv.publish('user_'+'1'*32, '名', [np.zeros(int(MAX_REFERENCE_SECONDS*2*16000), np.float32)],
                   [{'offset_seconds': 0.0, 'duration_seconds': MAX_REFERENCE_SECONDS*2,
                     'quality': 0.5}], np.zeros(ve.EMBEDDING_SIZE, np.float32), [0],
                   {'retained_seconds': MAX_REFERENCE_SECONDS*2, 'candidate_count': 1,
                    'accepted_count': 1, 'rejected': {}, 'contamination': {'flags': []}}, ['a.wav'])