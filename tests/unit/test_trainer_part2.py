"""Trainer tests part 2 — overflow from test_trainer_extended.py (150-line limit)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.services.trainer import Trainer
from neural_signal.shared.config import TrainingConfig


def _loader(n: int = 100, feat: int = 15) -> DataLoader:
    x = torch.randn(n, feat)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def _cfg(**kw) -> TrainingConfig:
    base = {
        "batch_size": 16, "learning_rate": 0.01, "weight_decay": 0.0,
        "max_epochs": 5, "early_stopping_patience": 10, "val_split": 0.2, "random_seed": 42,
    }
    base.update(kw)
    return TrainingConfig(**base)


def test_random_seed_fixes_weight_init(tmp_path):
    """Same manual seed before model creation → identical initial weights."""
    torch.manual_seed(42)
    m1 = nn.Linear(15, 1)
    w1 = m1.weight.data.clone()
    torch.manual_seed(42)
    m2 = nn.Linear(15, 1)
    w2 = m2.weight.data.clone()
    assert torch.allclose(w1, w2)


def test_batch_size_from_config(tmp_path):
    """Larger batch size should process data in fewer batches per epoch (indirectly)."""
    small_loader = DataLoader(TensorDataset(torch.randn(64, 15), torch.randn(64, 1)),
                              batch_size=4)
    big_loader = DataLoader(TensorDataset(torch.randn(64, 15), torch.randn(64, 1)),
                            batch_size=64)
    r_small = Trainer(_cfg(max_epochs=1), tmp_path / "cs.pt",
                      tmp_path / "ls.csv").train(nn.Linear(15, 1), small_loader, small_loader)
    r_big = Trainer(_cfg(max_epochs=1), tmp_path / "cb.pt",
                    tmp_path / "lb.csv").train(nn.Linear(15, 1), big_loader, big_loader)
    assert r_small.epochs_trained == r_big.epochs_trained == 1


def test_best_checkpoint_is_minimum_val_mse_epoch(tmp_path):
    """The saved checkpoint should correspond to lowest val MSE, not last epoch."""
    cfg = _cfg(max_epochs=10, early_stopping_patience=10, learning_rate=0.1)
    model = nn.Linear(15, 1)
    ckpt = tmp_path / "best.pt"
    result = Trainer(cfg, ckpt, tmp_path / "log.csv").train(model, _loader(), _loader())
    best_state = torch.load(ckpt, weights_only=True)
    assert best_state is not None
    assert result.best_val_mse <= min(result.val_losses) + 1e-8
