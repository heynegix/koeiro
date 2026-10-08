from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODEL_SPECS = {
    "tiny": dict(channels=64, hidden=80),
    "small": dict(channels=96, hidden=160),
    "medium": dict(channels=176, hidden=288),
}


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.kernel_size - 1, 0)))


class ProsodyNetS(nn.Module):
    """Causal Conv1D + GRU model with bounded pitch/energy corrections."""

    def __init__(self, input_dim: int = 8, channels: int = 96, hidden: int = 160):
        super().__init__()
        self.input_dim = int(input_dim)
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.conv1 = CausalConv1d(input_dim, channels)
        self.conv2 = CausalConv1d(channels, channels)
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def _bound(self, raw: torch.Tensor) -> torch.Tensor:
        pitch = torch.tanh(raw[..., 0:1]) * 1.2 + 0.2  # [-1.0, +1.4]
        energy = torch.tanh(raw[..., 1:2]) * 1.5       # [-1.5, +1.5]
        return torch.cat((pitch, energy), dim=-1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None,
                return_raw: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [batch,time,{self.input_dim}], got {tuple(x.shape)}")
        value = x.transpose(1, 2)
        value = F.relu(self.conv1(value))
        value = F.relu(self.conv2(value)).transpose(1, 2)
        value, hidden = self.gru(value, hidden)
        raw = self.head(value)
        return (raw if return_raw else self._bound(raw)), hidden

    def initial_state(self, batch_size: int = 1, device: torch.device | None = None) -> dict[str, Any]:
        device = device or next(self.parameters()).device
        return dict(conv1=torch.zeros(batch_size, self.input_dim, 2, device=device),
                    conv2=torch.zeros(batch_size, self.channels, 2, device=device),
                    hidden=torch.zeros(1, batch_size, self.hidden, device=device))

    @torch.no_grad()
    def step(self, frame: torch.Tensor, state: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        if frame.ndim == 1:
            frame = frame.unsqueeze(0)
        if frame.ndim != 2 or frame.shape[-1] != self.input_dim:
            raise ValueError("Streaming frame must be [batch,input_dim]")
        value = frame.unsqueeze(-1)
        input_history = torch.cat((state["conv1"], value), dim=-1)
        value = F.relu(self.conv1.conv(input_history))
        state["conv1"] = input_history[..., -2:].detach()
        channel_history = torch.cat((state["conv2"], value), dim=-1)
        value = F.relu(self.conv2.conv(channel_history))
        state["conv2"] = channel_history[..., -2:].detach()
        value = value.transpose(1, 2)
        value, hidden = self.gru(value, state["hidden"])
        state["hidden"] = hidden.detach()
        return self._bound(self.head(value[:, -1])), state

    def reset_state(self, state: dict[str, Any]) -> dict[str, Any]:
        for value in state.values():
            if isinstance(value, torch.Tensor):
                value.zero_()
        return state

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
