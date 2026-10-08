"""Deterministic teacher target generation for the v0.6 prosody dataset.

The realtime engine is deliberately not imported here.  This module works on
cached offline features and keeps pitch, energy and timing as separate targets.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .features import relative_f0, voiced_delta


def _finite_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def clean_f0(f0: np.ndarray, voiced: np.ndarray, *, min_island: int = 2,
             min_hz: float = 65.0, max_hz: float = 650.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove invalid frames and obvious isolated octave tracking errors."""
    raw = _finite_float(f0)
    v = np.asarray(voiced, dtype=np.float64).copy()
    out = np.where(np.isfinite(raw) & (raw >= min_hz) & (raw <= max_hz) & (v > 0), raw, 0.0)
    v[out == 0] = 0.0
    # Very short voiced islands are generally FCPE false positives in pauses.
    i = 0
    while i < len(out):
        if out[i] <= 0:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] > 0:
            j += 1
        if j - i < min_island:
            out[i:j] = 0.0
            v[i:j] = 0.0
        i = j
    corrected = 0
    # Correct a single octave spike only when both neighbours agree.  Genuine
    # rises are retained because their neighbours will not have the same Hz.
    for i in range(1, len(out) - 1):
        if out[i] <= 0 or out[i - 1] <= 0 or out[i + 1] <= 0:
            continue
        neighbour = (out[i - 1] + out[i + 1]) * 0.5
        ratio = out[i] / neighbour if neighbour else 1.0
        if (1.75 <= ratio <= 2.25) or (0.44 <= ratio <= 0.57):
            out[i] = neighbour
            corrected += 1
    return out.astype(np.float32), (out > 0).astype(np.float32), corrected


def robust_baseline(f0: np.ndarray, voiced: np.ndarray) -> float:
    values = _finite_float(f0)
    valid = (values > 0) & np.isfinite(values) & (np.asarray(voiced) > 0)
    return float(np.median(values[valid])) if np.any(valid) else 0.0


def relative_clean_f0(f0: np.ndarray, voiced: np.ndarray) -> tuple[np.ndarray, float]:
    clean, v, _ = clean_f0(f0, voiced)
    baseline = robust_baseline(clean, v)
    result = np.zeros(len(clean), dtype=np.float32)
    if baseline > 0:
        result[v > 0] = (12.0 * np.log2(clean[v > 0] / baseline)).astype(np.float32)
    return result, baseline


def _smooth(values: np.ndarray, valid: np.ndarray, passes: int = 2) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    valid = np.asarray(valid) > 0
    result[~valid] = 0.0
    for _ in range(passes):
        if len(result) < 3:
            break
        padded = np.pad(result, (1, 1), mode="edge")
        smooth = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
        result[valid] = smooth[valid]
    return result.astype(np.float32)


def _slew(values: np.ndarray, valid: np.ndarray, max_step: float = 0.42,
          max_accel: float = 0.24) -> np.ndarray:
    """Limit frame-to-frame movement (10 ms source frame)."""
    result = np.asarray(values, dtype=np.float64).copy()
    valid = np.asarray(valid) > 0
    previous = 0.0
    velocity = 0.0
    for i in range(len(result)):
        if not valid[i]:
            previous = 0.0
            velocity = 0.0
            result[i] = 0.0
            continue
        desired = result[i]
        step = float(np.clip(desired - previous, -max_step, max_step))
        step = float(np.clip(step, velocity - max_accel, velocity + max_accel))
        previous += step
        velocity = step
        result[i] = previous
    return result.astype(np.float32)


def _onset_lifts(voiced: np.ndarray, lift: float, min_silence_frames: int = 12) -> np.ndarray:
    v = np.asarray(voiced) > 0
    result = np.zeros(len(v), dtype=np.float32)
    silence = min_silence_frames
    for i, current in enumerate(v):
        if not current:
            silence += 1
            continue
        if silence >= min_silence_frames:
            width = min(15, len(v) - i)
            result[i:i + width] += np.linspace(lift, 0.0, width, dtype=np.float32)
        silence = 0
    return result


def _relative_energy(energy: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    e = np.asarray(energy, dtype=np.float64)
    v = np.asarray(voiced) > 0
    reference = float(np.median(e[v])) if np.any(v) else float(np.median(e))
    return (e - reference).astype(np.float32)


def rule_target(neutral: dict[str, Any], *, range_expansion: float = 1.35,
                rise_boost: float = 1.20, fall_boost: float = 1.05,
                onset_lift: float = 0.30, energy_scale: float = 1.15,
                pitch_clamp: tuple[float, float] = (-1.0, 1.4),
                energy_clamp: float = 1.5) -> dict[str, Any]:
    """Generate a bounded, smooth target while preserving neutral timing."""
    f0, voiced, corrected = clean_f0(neutral['f0'], neutral['voiced'])
    relative, baseline = relative_clean_f0(f0, voiced)
    movement = np.zeros(len(relative), dtype=np.float64)
    if len(relative) > 1:
        delta = np.diff(relative, prepend=relative[0]).astype(np.float64)
        movement = np.where(delta >= 0, delta * rise_boost, delta * fall_boost)
    # Expansion around the robust speaker baseline, plus gentle attack lift.
    correction = (relative.astype(np.float64) * (range_expansion - 1.0))
    correction += movement * 0.10
    correction += _onset_lifts(voiced, onset_lift).astype(np.float64)
    correction = np.clip(correction, pitch_clamp[0], pitch_clamp[1])
    correction = _slew(_smooth(correction, voiced), voiced)
    correction = np.clip(correction, pitch_clamp[0], pitch_clamp[1]).astype(np.float32)
    target_relative = (relative + correction).astype(np.float32)

    neutral_energy = np.asarray(neutral['energy'], dtype=np.float32)
    relative_energy = _relative_energy(neutral_energy, voiced)
    energy_delta = np.clip(relative_energy * (energy_scale - 1.0), -energy_clamp, energy_clamp)
    target_energy = neutral_energy + energy_delta
    return dict(
        neutral_f0=f0, neutral_voiced=voiced, neutral_relative_f0_st=relative,
        baseline_f0=np.float32(baseline), target_relative_f0_st=target_relative,
        target_delta_f0_st=correction, neutral_energy=neutral_energy,
        neutral_relative_energy=relative_energy, target_energy=target_energy.astype(np.float32),
        target_delta_energy=energy_delta.astype(np.float32), target_voiced=voiced,
        time=np.asarray(neutral['time'], dtype=np.float64),
        duration=np.float64(neutral['duration']), hop_seconds=np.float64(neutral['hop_seconds']),
        f0_corrections=np.int32(corrected), teacher_type=np.array('rule'),
    )


def _map_values(values: np.ndarray, mapping: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mapping = np.clip(np.asarray(mapping, dtype=np.int32), 0, len(values) - 1)
    if len(mapping) == length:
        return values[mapping]
    return np.interp(np.arange(length), np.linspace(0, length - 1, len(values)), values).astype(np.float32)


def hybrid_target(neutral: dict[str, Any], anime: dict[str, Any], mapping: np.ndarray,
                  *, energy_blend: float = 0.75) -> dict[str, Any]:
    """Rule pitch + MioTTS anime energy/timing, aligned to neutral frames."""
    target = rule_target(neutral)
    anime_energy = _relative_energy(anime['energy'], anime['voiced'])
    aligned_energy = _map_values(anime_energy, mapping, len(target['target_energy']))
    rule_energy = np.asarray(target['neutral_relative_energy'], dtype=np.float32)
    delta = np.clip(energy_blend * aligned_energy + (1.0 - energy_blend) * rule_energy - rule_energy,
                    -1.5, 1.5).astype(np.float32)
    target['target_energy'] = (np.asarray(neutral['energy'], dtype=np.float32) + delta).astype(np.float32)
    target['target_delta_energy'] = delta
    target['teacher_type'] = np.array('hybrid')
    target['source_anime_energy'] = aligned_energy
    return target


def corrected_anime_target(neutral: dict[str, Any], anime: dict[str, Any], mapping: np.ndarray) -> dict[str, Any]:
    """MioTTS energy/timing with rule expansion only when pitch is flat."""
    result = hybrid_target(neutral, anime, mapping)
    anime_rel, _ = relative_clean_f0(anime['f0'], anime['voiced'])
    aligned = _map_values(anime_rel, mapping, len(result['target_relative_f0_st']))
    nrel = result['neutral_relative_f0_st']
    ratio = float(np.std(aligned) / max(np.std(nrel), 1e-6))
    if ratio < 1.10:
        result['target_delta_f0_st'] = np.clip(result['target_delta_f0_st'] * (1.10 / max(ratio, .25)), -1.0, 1.4)
        result['target_relative_f0_st'] = (nrel + result['target_delta_f0_st']).astype(np.float32)
    result['teacher_type'] = np.array('mio_rule_correction')
    return result


def _distribution(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(mean=0., std=0., iqr=0., range=0.)
    return dict(mean=float(np.mean(x)), std=float(np.std(x)),
                iqr=float(np.percentile(x, 75) - np.percentile(x, 25)),
                range=float(np.max(x) - np.min(x)))


def teacher_scores(neutral: dict[str, Any], target: dict[str, Any], alignment: dict[str, Any] | None = None) -> dict[str, Any]:
    nv = np.asarray(target.get('neutral_voiced', neutral['voiced'])) > 0
    nrel = np.asarray(target.get('neutral_relative_f0_st', relative_f0(neutral['f0'])), dtype=np.float32)
    trel = np.asarray(target['target_relative_f0_st'], dtype=np.float32)
    ne = _relative_energy(np.asarray(neutral['energy']), nv)
    te = np.asarray(target['target_energy'], dtype=np.float32)
    td = np.asarray(target['target_delta_f0_st'], dtype=np.float32)
    nd = voiced_delta(nrel, nv)
    tv = np.asarray(target.get('target_voiced', nv)) > 0
    nstats, tstats = _distribution(nrel[nv]), _distribution(trel[tv])
    nenergy, tenergy = _distribution(ne[nv]), _distribution((te - np.median(te[nv]) if np.any(nv) else te))
    npos = nd[nd > 0]; tpos = td[td > 0]; nneg = nd[nd < 0]; tneg = td[td < 0]
    std_ratio = tstats['std'] / max(nstats['std'], 1e-6)
    range_ratio = tstats['range'] / max(nstats['range'], 1e-6)
    iqr_ratio = tstats['iqr'] / max(nstats['iqr'], 1e-6)
    energy_ratio = tenergy['std'] / max(nenergy['std'], 1e-6)
    duration_ratio = float(target.get('duration', neutral['duration'])) / max(float(neutral['duration']), 1e-6)
    alignment_score = 1.0 if alignment is None else float(np.clip(1.0 - alignment.get('mean_cost', 1.0) / 3.0, 0, 1))
    pitch_score = float(np.clip(.5 * min(std_ratio / 1.15, 1.0) + .5 * min(range_ratio / 1.10, 1.0), 0, 1))
    energy_score = float(np.clip(min(energy_ratio / 1.05, 1.0), 0, 1))
    timing_score = float(np.clip(1.0 - abs(math.log(max(duration_ratio, 1e-6))) / math.log(2), 0, 1))
    validity = float(np.isfinite(trel).all() and np.isfinite(te).all() and np.max(np.abs(td)) <= 1.5001)
    overall = float(.40 * pitch_score + .20 * energy_score + .15 * timing_score + .20 * alignment_score + .05 * validity)
    return dict(f0_std_ratio=float(std_ratio), f0_range_ratio=float(range_ratio), f0_iqr_ratio=float(iqr_ratio),
                energy_std_ratio=float(energy_ratio), energy_range_ratio=float(tenergy['range'] / max(nenergy['range'], 1e-6)),
                positive_velocity=float(np.mean(tpos) if len(tpos) else 0.), negative_velocity=float(np.mean(np.abs(tneg)) if len(tneg) else 0.),
                neutral_positive_velocity=float(np.mean(npos) if len(npos) else 0.), neutral_negative_velocity=float(np.mean(np.abs(nneg)) if len(nneg) else 0.),
                duration_ratio=duration_ratio, voiced_ratio=float(np.mean(tv)), alignment_score=alignment_score,
                pitch_score=pitch_score, energy_score=energy_score, timing_score=timing_score,
                validity_score=validity, overall_score=overall)


def target_sample(neutral: dict[str, Any], target: dict[str, Any], mapping: np.ndarray,
                  scores: dict[str, Any], sentence_id: str) -> dict[str, Any]:
    """Return the stable v0.7 training schema."""
    result = dict(sentence_id=np.array(sentence_id),
                  neutral_f0_st=np.asarray(target['neutral_relative_f0_st'], dtype=np.float32),
                  neutral_f0_delta=voiced_delta(target['neutral_relative_f0_st'], target['neutral_voiced']),
                  neutral_energy=np.asarray(target['neutral_relative_energy'], dtype=np.float32),
                  neutral_energy_delta=np.diff(target['neutral_relative_energy'], prepend=target['neutral_relative_energy'][0]).astype(np.float32),
                  neutral_voiced=np.asarray(target['neutral_voiced'], dtype=np.float32),
                  target_f0_st=np.asarray(target['target_relative_f0_st'], dtype=np.float32),
                  target_delta_f0_st=np.asarray(target['target_delta_f0_st'], dtype=np.float32),
                  target_energy=(np.asarray(target['neutral_relative_energy'], dtype=np.float32) +
                                 np.asarray(target['target_delta_energy'], dtype=np.float32)),
                  target_delta_energy=np.asarray(target['target_delta_energy'], dtype=np.float32),
                  target_voiced=np.asarray(target.get('target_voiced', target['neutral_voiced']), dtype=np.float32),
                  timing=np.asarray(target['time'], dtype=np.float64), alignment=np.asarray(mapping, dtype=np.int32),
                  quality_scores=np.array(str(scores), dtype='<U2048'), frame_ms=np.float32(float(target['hop_seconds']) * 1000), schema_version=np.int32(2))
    return result
