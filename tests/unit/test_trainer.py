"""Tests for services/trainer.py — Trainer training loop."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.services.trainer import Trainer, TrainingResult
from neural_signal.shared.config import TrainingConfig


@pytest.fixture()
def train_cfg() -> TrainingConfig:
    return TrainingConfig(
        batch_size=16,
        learning_rate=0.01,
        weight_decay=0.0,
        max_epochs=5,
        early_stopping_patience=3,
        val_split=0.2,
        random_seed=42,
    )


@pytest.fixture()
def simple_model() -> nn.Module:
    return nn.Linear(15, 1)


def _make_loader(n: int = 100, feat: int = 15) -> DataLoader:
    x = torch.randn(n, feat)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def test_trainer_instantiates(train_cfg, tmp_path):
    t = Trainer(train_cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv")
    assert t is not None


def test_train_returns_training_result(train_cfg, simple_model, tmp_path):
    loader = _make_loader()
    t = Trainer(train_cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv")
    result = t.train(simple_model, loader, loader)
    assert isinstance(result, TrainingResult)


def test_train_result_epochs_trained_positive(train_cfg, simple_model, tmp_path):
    loader = _make_loader()
    result = Trainer(train_cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        simple_model, loader, loader
    )
    assert result.epochs_trained > 0


def test_train_result_best_val_mse_finite(train_cfg, simple_model, tmp_path):
    loader = _make_loader()
    result = Trainer(train_cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        simple_model, loader, loader
    )
    assert result.best_val_mse < float("inf")


def test_checkpoint_saved_after_training(train_cfg, simple_model, tmp_path):
    loader = _make_loader()
    ckpt = tmp_path / "best.pt"
    Trainer(train_cfg, ckpt, tmp_path / "log.csv").train(simple_model, loader, loader)
    assert ckpt.exists()


def test_training_log_csv_saved(train_cfg, simple_model, tmp_path):
    loader = _make_loader()
    log = tmp_path / "log.csv"
    Trainer(train_cfg, tmp_path / "ckpt.pt", log).train(simple_model, loader, loader)
    assert log.exists()


def test_train_losses_list_length_equals_epochs(train_cfg, simple_model, tmp_path):
    loader = _make_loader()
    result = Trainer(train_cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        simple_model, loader, loader
    )
    assert len(result.train_losses) == result.epochs_trained


def test_val_losses_list_length_equals_epochs(train_cfg, simple_model, tmp_path):
    loader = _make_loader()
    result = Trainer(train_cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        simple_model, loader, loader
    )
    assert len(result.val_losses) == result.epochs_trained


def test_early_stopping_fires(tmp_path):
    """Model that never improves should stop at patience."""
    cfg = TrainingConfig(
        batch_size=16, learning_rate=0.0, weight_decay=0.0,
        max_epochs=50, early_stopping_patience=2,
        val_split=0.2, random_seed=42,
    )
    model = nn.Linear(15, 1)
    loader = _make_loader()
    result = Trainer(cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        model, loader, loader
    )
    assert result.stopped_early


def test_no_nan_in_train_losses(train_cfg, simple_model, tmp_path):
    loader = _make_loader()
    result = Trainer(train_cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        simple_model, loader, loader
    )
    assert all(v == v for v in result.train_losses)  # NaN check: NaN != NaN
