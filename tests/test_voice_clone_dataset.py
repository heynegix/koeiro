from pathlib import Path

import numpy as np

from tools.build_voice_clone_dataset import (
    DEFAULT_SR,
    MAX_CLIP_SEC,
    MIN_CLIP_SEC,
    detect_speech_regions,
    estimate_f0,
    read_wav,
    validate_dataset,
    write_wav,
)


def test_detect_speech_regions_keeps_separated_tones():
    sr = DEFAULT_SR
    t = np.arange(sr * 6, dtype=np.float32) / sr
    audio = np.zeros_like(t)
    audio[int(0.5 * sr) : int(2.7 * sr)] = 0.15 * np.sin(2 * np.pi * 220 * t[int(0.5 * sr) : int(2.7 * sr)])
    audio[int(3.2 * sr) : int(5.4 * sr)] = 0.15 * np.sin(2 * np.pi * 250 * t[int(3.2 * sr) : int(5.4 * sr)])
    regions = detect_speech_regions(audio, sr)
    assert len(regions) == 2
    assert all(1.8 <= (end - start) / sr <= 2.6 for start, end in regions)


def test_estimate_f0_is_finite_for_voiced_sine():
    sr = DEFAULT_SR
    t = np.arange(sr * 1.0, dtype=np.float32) / sr
    f0, voiced = estimate_f0(0.2 * np.sin(2 * np.pi * 220 * t), sr)
    voiced_values = f0[f0 > 0]
    assert voiced > 0.5
    assert len(voiced_values) > 0
    assert np.isfinite(voiced_values).all()
    assert abs(float(np.median(voiced_values)) - 220.0) < 8.0


def test_wav_round_trip_is_mono_48k_and_bounded(tmp_path: Path):
    sr = DEFAULT_SR
    t = np.arange(sr, dtype=np.float32) / sr
    source = 0.2 * np.sin(2 * np.pi * 180 * t)
    path = tmp_path / "clip.wav"
    write_wav(path, source, sr)
    restored, rate, channels = read_wav(path)
    assert rate == sr
    assert channels == 1
    assert len(restored) / rate == 1.0
    assert np.max(np.abs(restored)) < 1.0


def test_validator_rejects_missing_or_bad_duration(tmp_path: Path):
    (tmp_path / "wav").mkdir()
    metadata = [
        {
            "id": "clip_0001",
            "status": "accepted",
            "wav_path": "wav/missing.wav",
        }
    ]
    report = validate_dataset(tmp_path, metadata, DEFAULT_SR)
    assert report["passed"] is False
    assert report["failure_count"] == 1
    assert MIN_CLIP_SEC < MAX_CLIP_SEC
