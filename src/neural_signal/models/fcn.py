"""FCNModel — Fully-Connected Network for signal regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_signal import constants as const
from neural_signal.shared.config import FCNConfig


class FCNModel(nn.Module):
    """Dense network: Linear(15→h0,ReLU,Drop) → Linear(h0→h1,ReLU,Drop) → Linear(h1→1).

    All architecture sizes come from FCNConfig — no literals in this file.
    L2 regularisation is applied via weight_decay in the Adam optimiser.
    """

    def __init__(self, cfg: FCNConfig) -> None:
        """Build layers from FCNConfig (hidden_sizes, dropout_rate, output_size)."""
        super().__init__()
        h0, h1 = cfg.hidden_sizes[0], cfg.hidden_sizes[1]
        self.network = nn.Sequential(
            nn.Linear(const.INPUT_SIZE_FCN, h0),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(h0, h1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(h1, cfg.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass. Input shape: (batch, 15). Output shape: (batch, 1)."""
        return self.network(x)
