import numpy as np

from src.dataset.teachers import (
    clean_f0, hybrid_target, rule_target, teacher_scores, target_sample,
)


def synthetic(length=160):
    time = np.arange(length, dtype=np.float64) * .01
    f0 = 190.0 + 22.0 * np.sin(time * 3.0)
    voiced = np.ones(length, dtype=np.float32)
    voiced[:10] = 0
    voiced[-8:] = 0
    energy = -30.0 + 3.0 * np.sin(time * 2.0)
    return dict(f0=f0.astype(np.float32), voiced=voiced, energy=energy.astype(np.float32),
                time=time, duration=np.float64(length * .01), hop_seconds=np.float64(.01))


def test_clean_f0_removes_invalid_and_octave_spike():
    f = np.full(30, 200.0, dtype=np.float32)
    v = np.ones(30, dtype=np.float32)
    f[15] = 400.0
    f[0] = np.nan
    cleaned, valid, corrections = clean_f0(f, v)
    assert cleaned[0] == 0 and valid[0] == 0
    assert abs(float(cleaned[15]) - 200.0) < 1
    assert corrections == 1


def test_rule_target_has_finite_bounded_delta_and_same_length():
    target = rule_target(synthetic())
    assert len(target["target_delta_f0_st"]) == len(target["neutral_relative_f0_st"])
    assert np.isfinite(target["target_relative_f0_st"]).all()
    assert np.isfinite(target["target_energy"]).all()
    assert np.min(target["target_delta_f0_st"]) >= -1.0001
    assert np.max(target["target_delta_f0_st"]) <= 1.4001


def test_hybrid_uses_anime_energy_but_rule_pitch():
    n = synthetic()
    a = synthetic()
    a["energy"] = a["energy"] + 8.0 * np.sin(a["time"] * 5.0)
    mapping = np.arange(len(n["f0"]), dtype=np.int32)
    target = hybrid_target(n, a, mapping)
    rule = rule_target(n)
    assert np.allclose(target["target_delta_f0_st"], rule["target_delta_f0_st"])
    assert np.max(np.abs(target["target_delta_energy"])) <= 1.5 + 1e-6


def test_teacher_scores_detect_pitch_expansion():
    n = synthetic()
    target = rule_target(n)
    scores = teacher_scores(n, target)
    assert scores["f0_std_ratio"] > 1.0
    assert scores["f0_range_ratio"] > 1.0
    assert scores["validity_score"] == 1.0


def test_training_schema_and_alignment():
    n = synthetic(80)
    target = rule_target(n)
    mapping = np.arange(80, dtype=np.int32)
    scores = teacher_scores(n, target)
    sample = target_sample(n, target, mapping, scores, "000001")
    for key in ("neutral_f0_st", "neutral_f0_delta", "neutral_energy", "target_f0_st", "target_delta_f0_st", "target_energy", "target_delta_energy", "alignment"):
        assert key in sample
    assert sample["target_f0_st"].shape == sample["alignment"].shape
