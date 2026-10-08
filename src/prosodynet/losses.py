from __future__ import annotations

import torch
from torch.nn import functional as F


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def prosody_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 voiced: torch.Tensor, weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or dict(pitch=1.0, energy=.30, velocity=.20, smoothness=.05)
    valid_pitch = mask * voiced
    pitch = _masked_mean(F.smooth_l1_loss(pred[..., 0], target[..., 0], reduction="none"), valid_pitch)
    energy = _masked_mean(F.smooth_l1_loss(pred[..., 1], target[..., 1], reduction="none"), mask)
    if pred.shape[1] > 1:
        adjacent = mask[:, 1:] * mask[:, :-1] * voiced[:, 1:] * voiced[:, :-1]
        velocity = _masked_mean(F.smooth_l1_loss(pred[:, 1:, 0] - pred[:, :-1, 0],
                                                 target[:, 1:, 0] - target[:, :-1, 0], reduction="none"), adjacent)
    else:
        velocity = pred.new_zeros(())
    if pred.shape[1] > 2:
        second_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2] * voiced[:, 2:] * voiced[:, 1:-1] * voiced[:, :-2]
        second = pred[:, 2:, 0] - 2 * pred[:, 1:-1, 0] + pred[:, :-2, 0]
        smoothness = _masked_mean(second.abs(), second_mask)
    else:
        smoothness = pred.new_zeros(())
    total = weights["pitch"] * pitch + weights["energy"] * energy + weights["velocity"] * velocity + weights["smoothness"] * smoothness
    return total, dict(pitch=float(pitch.detach()), energy=float(energy.detach()), velocity=float(velocity.detach()), smoothness=float(smoothness.detach()), total=float(total.detach()))
