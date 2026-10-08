"""Standard sinusoidal encoding, preserving the checkpoint's `pe` buffer key.

Independent minimal implementation: no SpeechBrain/ASR runtime dependency.
"""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, input_size, max_len=2500):
        super().__init__()
        if input_size % 2:
            raise ValueError('Encoding requires an even width')
        position = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        frequency = torch.exp(torch.arange(0, input_size, 2, dtype=torch.float32)
                              * (-math.log(10000)/input_size))
        table = torch.empty(max_len, input_size)
        table[:, 0::2] = torch.sin(position*frequency)
        table[:, 1::2] = torch.cos(position*frequency)
        self.register_buffer('pe', table.unsqueeze(0))

    def forward(self, audio):
        return self.pe[:, :audio.shape[1]].clone().detach()
