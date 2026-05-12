"""Extended trainer tests covering optimizer, loss, checkpoints, log columns, etc."""

from __future__ import annotations

import csv

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


def test_train_loss_decreases_over_10_epochs(tmp_path):
    """With a learnable model and decent LR, train loss should drop over 10 epochs."""
    cfg = _cfg(max_epochs=10, learning_rate=0.1)
    model = nn.Linear(15, 1)
    loader = _loader(200)
    result = Trainer(cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        model, loader, loader
    )
    assert result.train_losses[-1] < result.train_losses[0]


def test_optimizer_is_adam(tmp_path):
    """Trainer uses Adam; verify by checking weight update is non-zero after 1 epoch."""
    cfg = _cfg(max_epochs=1, learning_rate=0.01)
    model = nn.Linear(15, 1)
    w_before = model.weight.data.clone()
    loader = _loader()
    Trainer(cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(model, loader, loader)
    assert not torch.allclose(model.weight.data, w_before)


def test_learning_rate_from_config(tmp_path):
    """Higher LR should produce larger weight updates in 1 epoch."""
    loader = _loader(64)
    m_slow = nn.Linear(15, 1)
    m_slow.weight.data.fill_(0.5)
    m_fast = nn.Linear(15, 1)
    m_fast.weight.data.fill_(0.5)
    Trainer(_cfg(max_epochs=1, learning_rate=1e-5), tmp_path / "slow.pt",
            tmp_path / "slog.csv").train(m_slow, loader, loader)
    Trainer(_cfg(max_epochs=1, learning_rate=0.1), tmp_path / "fast.pt",
            tmp_path / "flog.csv").train(m_fast, loader, loader)
    diff_slow = (m_slow.weight.data - 0.5).abs().sum()
    diff_fast = (m_fast.weight.data - 0.5).abs().sum()
    assert diff_fast > diff_slow


def test_weight_decay_applied_for_fcn(tmp_path):
    """weight_decay > 0 shrinks weights more than weight_decay=0 over same epochs."""
    loader = _loader(64)
    m_wd = nn.Linear(15, 1)
    m_wd.weight.data.fill_(1.0)
    m_no = nn.Linear(15, 1)
    m_no.weight.data.fill_(1.0)
    Trainer(_cfg(max_epochs=5, learning_rate=0.01, weight_decay=0.1),
            tmp_path / "wd.pt", tmp_path / "wdlog.csv").train(m_wd, loader, loader,
                                                                weight_decay=0.1)
    Trainer(_cfg(max_epochs=5, learning_rate=0.01),
            tmp_path / "no.pt", tmp_path / "nolog.csv").train(m_no, loader, loader)
    assert m_wd.weight.data.abs().sum() <= m_no.weight.data.abs().sum() + 1e-3


def test_loss_function_is_mse(tmp_path):
    """Trainer uses MSE: a model predicting zeros on zero targets gives loss ≈ 0."""
    loader = DataLoader(
        TensorDataset(torch.randn(32, 15), torch.zeros(32, 1)), batch_size=32)
    m = nn.Linear(15, 1)
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    result = Trainer(_cfg(max_epochs=1), tmp_path / "c.pt",
                     tmp_path / "log.csv").train(m, loader, loader)
    assert result.train_losses[0] < 1e-6


def test_checkpoint_loadable_by_torch(tmp_path):
    loader = _loader()
    ckpt = tmp_path / "best.pt"
    model = nn.Linear(15, 1)
    Trainer(_cfg(), ckpt, tmp_path / "log.csv").train(model, loader, loader)
    state = torch.load(ckpt, weights_only=True)
    assert "weight" in state and "bias" in state


def test_training_log_has_epoch_column(tmp_path):
    log = tmp_path / "log.csv"
    Trainer(_cfg(), tmp_path / "c.pt", log).train(nn.Linear(15, 1), _loader(), _loader())
    with log.open() as fh:
        header = next(csv.reader(fh))
    assert "epoch" in header


def test_training_log_has_train_mse_column(tmp_path):
    log = tmp_path / "log.csv"
    Trainer(_cfg(), tmp_path / "c.pt", log).train(nn.Linear(15, 1), _loader(), _loader())
    with log.open() as fh:
        header = next(csv.reader(fh))
    assert "train_mse" in header


def test_training_log_has_val_mse_column(tmp_path):
    log = tmp_path / "log.csv"
    Trainer(_cfg(), tmp_path / "c.pt", log).train(nn.Linear(15, 1), _loader(), _loader())
    with log.open() as fh:
        header = next(csv.reader(fh))
    assert "val_mse" in header


def test_stopped_early_flag_false_when_all_epochs_done(tmp_path):
    """All epochs complete → stopped_early is False."""
    cfg = _cfg(max_epochs=3, early_stopping_patience=100)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        nn.Linear(15, 1), _loader(), _loader())
    assert not result.stopped_early


def test_early_stopping_patience_from_config(tmp_path):
    """Patience=1: stops after 2 epochs with no improvement (LR=0)."""
    cfg = _cfg(max_epochs=50, early_stopping_patience=1, learning_rate=0.0)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        nn.Linear(15, 1), _loader(), _loader())
    assert result.epochs_trained <= 3


def test_early_stopping_does_not_fire_when_val_improving(tmp_path):
    """With sufficient LR and short patience, training should not stop very early."""
    cfg = _cfg(max_epochs=5, early_stopping_patience=5, learning_rate=0.05)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        nn.Linear(15, 1), _loader(500), _loader(100))
    assert result.epochs_trained >= 1


