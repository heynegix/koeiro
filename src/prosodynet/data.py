from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


INPUT_FEATURES = (
    "relative_f0_st", "delta_f0_st", "energy_db", "delta_energy_db",
    "voiced", "speech_position", "time_since_onset", "recent_voiced_ratio",
)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def load_splits(root: Path) -> dict[str, list[str]]:
    data = json.loads((Path(root) / "splits" / "splits.json").read_text(encoding="utf-8"))
    splits = {name: list(data[name]) for name in ("train", "validation", "test")}
    ids = [item for values in splits.values() for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset leakage: sentence ID occurs in more than one split")
    if not ids:
        raise ValueError("Empty dataset split")
    return splits


def load_sample(root: Path, identifier: str) -> dict[str, np.ndarray]:
    path = Path(root) / "samples" / f"{identifier}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        sample = {key: data[key] for key in data.files}
    required = ("neutral_f0_st", "neutral_f0_delta", "neutral_energy", "neutral_energy_delta",
                "neutral_voiced", "target_delta_f0_st", "target_delta_energy")
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"Sample {identifier} missing fields: {missing}")
    length = len(sample["neutral_f0_st"])
    if any(len(sample[key]) != length for key in required):
        raise ValueError(f"Sample {identifier} has mismatched frame lengths")
    if length == 0:
        raise ValueError(f"Sample {identifier} is empty")
    if not all(np.isfinite(sample[key]).all() for key in required):
        raise ValueError(f"Sample {identifier} contains NaN/Inf")
    return sample


def causal_features(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Build features using only the current and preceding frames."""
    length = len(sample["neutral_f0_st"])
    f0 = _finite(sample["neutral_f0_st"])
    f0_delta = _finite(sample["neutral_f0_delta"])
    energy = _finite(sample["neutral_energy"])
    energy_delta = _finite(sample["neutral_energy_delta"])
    voiced = (_finite(sample["neutral_voiced"]) > 0).astype(np.float32)
    position = np.linspace(0.0, 1.0, length, dtype=np.float32)
    # The first voiced frame is only known after it arrives.  Before that
    # frame, time_since_onset remains zero; no future frame is inspected.
    time_since_onset = np.zeros(length, dtype=np.float32)
    recent_ratio = np.zeros(length, dtype=np.float32)
    onset = None
    window = 50  # 500 ms at the Dataset v2 10 ms frame rate.
    for index in range(length):
        if voiced[index] > 0 and onset is None:
            onset = index
        if onset is not None:
            time_since_onset[index] = min(1.0, max(0.0, (index - onset) / window))
        start = max(0, index - window + 1)
        recent_ratio[index] = float(np.mean(voiced[start:index + 1]))
    result = np.column_stack((f0, f0_delta, energy, energy_delta, voiced,
                              position, time_since_onset, recent_ratio)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite causal features")
    return result


def targets(sample: dict[str, np.ndarray]) -> np.ndarray:
    result = np.column_stack((_finite(sample["target_delta_f0_st"]),
                              _finite(sample["target_delta_energy"]))).astype(np.float32)
    return result


def training_statistics(root: Path, train_ids: list[str]) -> dict[str, Any]:
    values = np.concatenate([causal_features(load_sample(root, identifier)) for identifier in train_ids], axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return dict(features=list(INPUT_FEATURES), mean=mean.astype(float).tolist(), std=std.astype(float).tolist(),
                min=values.min(axis=0).astype(float).tolist(), max=values.max(axis=0).astype(float).tolist(),
                frame_ms=10, history_ms=500, source_split="train only", count=int(len(values)))


def normalize(features: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((features - mean) / std).astype(np.float32)


class WindowDataset:
    """Fixed causal windows with explicit padding masks."""

    def __init__(self, root: Path, identifiers: list[str], stats: dict[str, Any], sequence_length: int = 50):
        self.root = Path(root)
        self.identifiers = list(identifiers)
        self.stats = stats
        self.sequence_length = int(sequence_length)
        # The training set is small enough to keep decoded NPZ arrays in
        # memory.  Avoid reopening the same file for every window and epoch.
        self.cache = {identifier: load_sample(self.root, identifier) for identifier in self.identifiers}
        self.processed = {identifier: (normalize(causal_features(sample), self.stats), targets(sample))
                          for identifier, sample in self.cache.items()}
        self.windows: list[tuple[str, int]] = []
        for identifier in self.identifiers:
            length = len(self.cache[identifier]["neutral_f0_st"])
            self.windows.extend((identifier, start) for start in range(0, length, self.sequence_length))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        identifier, start = self.windows[index]
        sample = self.cache[identifier]
        features, labels = self.processed[identifier]
        end = min(start + self.sequence_length, len(features))
        length = end - start
        x = np.zeros((self.sequence_length, features.shape[1]), dtype=np.float32)
        y = np.zeros((self.sequence_length, labels.shape[1]), dtype=np.float32)
        mask = np.zeros(self.sequence_length, dtype=np.float32)
        voiced = np.zeros(self.sequence_length, dtype=np.float32)
        x[:length] = features[start:end]
        y[:length] = labels[start:end]
        mask[:length] = 1.0
        voiced[:length] = sample["neutral_voiced"][start:end] > 0
        return dict(x=x, y=y, mask=mask, voiced=voiced, id=identifier, start=start)


def split_leakage(splits: dict[str, list[str]]) -> list[tuple[str, str, list[str]]]:
    problems = []
    names = ("train", "validation", "test")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = sorted(set(splits[left]) & set(splits[right]))
            if overlap:
                problems.append((left, right, overlap))
    return problems
