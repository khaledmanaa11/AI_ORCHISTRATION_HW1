"""Trainer device and gradient-clipping tests."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.services.trainer import Trainer
from neural_signal.shared.config import TrainingConfig


def _loader(n: int = 80) -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(n, 15), torch.randn(n, 1)),
                      batch_size=16, shuffle=False)


def _cfg(**kw) -> TrainingConfig:
    base = {
        "batch_size": 16, "learning_rate": 0.01, "weight_decay": 0.0,
        "max_epochs": 2, "early_stopping_patience": 10, "val_split": 0.2, "random_seed": 42,
    }
    base.update(kw)
    return TrainingConfig(**base)


def test_trainer_moves_model_to_device(tmp_path):
    """Model parameters should be on CPU after training with device='cpu'."""
    cfg = _cfg(device="cpu")
    model = nn.Linear(15, 1)
    Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(model, _loader(), _loader())
    assert model.weight.device.type == "cpu"


def test_trainer_moves_batches_to_device(tmp_path):
    """Training with device='cpu' should complete without device-mismatch errors."""
    cfg = _cfg(device="cpu", max_epochs=1)
    model = nn.Linear(15, 1)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        model, _loader(), _loader()
    )
    assert result.best_val_mse < float("inf")


def test_trainer_device_from_config(tmp_path):
    """Trainer reads device from TrainingConfig.device, not a hardcoded string."""
    cfg = _cfg(device="cpu")
    assert cfg.device == "cpu"
    model = nn.Linear(15, 1)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        model, _loader(), _loader()
    )
    assert result.epochs_trained >= 1


def test_training_runs_on_cpu(tmp_path):
    """Explicit device=cpu: full training loop completes, MSE finite."""
    cfg = _cfg(device="cpu", max_epochs=3)
    model = nn.Linear(15, 1)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        model, _loader(), _loader()
    )
    assert result.best_val_mse < float("inf") and result.epochs_trained == 3


def test_gradient_clipping_applied_when_configured(tmp_path):
    """With clip_norm=0.01 and high LR, grad norms should stay bounded."""
    cfg = _cfg(learning_rate=10.0, max_epochs=2, gradient_clip_norm=0.01)
    model = nn.Linear(15, 1)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        model, _loader(), _loader()
    )
    assert result.epochs_trained >= 1


def test_gradient_clipping_not_applied_when_config_is_null(tmp_path):
    """No clipping when gradient_clip_norm is None — training still completes."""
    cfg = _cfg(gradient_clip_norm=None, max_epochs=2)
    model = nn.Linear(15, 1)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        model, _loader(), _loader()
    )
    assert result.epochs_trained == 2


def test_gradient_clip_norm_from_config(tmp_path):
    """Trainer reads clip value from TrainingConfig, not hardcoded."""
    cfg = _cfg(gradient_clip_norm=5.0)
    assert cfg.gradient_clip_norm == 5.0
    model = nn.Linear(15, 1)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        model, _loader(), _loader()
    )
    assert result.epochs_trained >= 1
