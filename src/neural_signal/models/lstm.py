"""LSTMModel — LSTM with dense head for many-to-one signal regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_signal.shared.config import LSTMConfig


class LSTMModel(nn.Module):
    """LSTM: LSTM(input=cfg.input_size, hidden=64) → Linear(64→32) → ReLU → Linear(32→1).

    Uses the last time-step output. All sizes from LSTMConfig — no literals here.
    """

    def __init__(self, cfg: LSTMConfig) -> None:
        """Build LSTM and dense head from LSTMConfig."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.dense_hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.dense_hidden_size, cfg.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass. Input: (batch, seq_len, 1). Output: (batch, 1)."""
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
