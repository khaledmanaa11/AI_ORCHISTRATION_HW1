"""RNNModel — Vanilla RNN for many-to-one signal regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_signal.shared.config import RNNConfig


class RNNModel(nn.Module):
    """Vanilla RNN: RNN(input=1, hidden=64, tanh) → Linear(64→1), many-to-one.

    Extracts the last hidden state output[:, -1, :] and maps it to a scalar.
    All sizes from RNNConfig — no literals here.
    """

    def __init__(self, cfg: RNNConfig) -> None:
        """Build RNN and linear head from RNNConfig."""
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            nonlinearity=cfg.nonlinearity,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass. Input shape: (batch, seq_len, 1). Output: (batch, 1)."""
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])
