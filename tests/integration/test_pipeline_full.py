"""Phase 23 + Gap 11 integration tests — full pipeline and val-MSE divergence."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.models.fcn import FCNModel
from neural_signal.models.lstm import LSTMModel
from neural_signal.models.rnn import RNNModel
from neural_signal.services.evaluator import Evaluator
from neural_signal.services.trainer import Trainer, TrainingResult
from neural_signal.shared.config import FCNConfig, LSTMConfig, RNNConfig, TrainingConfig


def _fcn_loader(n: int = 120) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.randn(n, 15), torch.randn(n, 1)),
        batch_size=32, shuffle=False,
    )


def _seq_loader(n: int = 120) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.randn(n, 10, 1), torch.randn(n, 1)),
        batch_size=32, shuffle=False,
    )


def _fast_cfg(**overrides) -> TrainingConfig:
    base = {
        "batch_size": 32, "learning_rate": 0.01, "weight_decay": 0.0,
        "max_epochs": 5, "early_stopping_patience": 5, "val_split": 0.2, "random_seed": 42,
    }
    base.update(overrides)
    return TrainingConfig(**base)


# ── Phase 23: full train→evaluate integration ─────────────────────────────────

def test_fcn_train_then_cold_evaluate(tmp_path):
    """Train FCN, reload from checkpoint (cold load), then evaluate — MSE must be finite."""
    cfg = _fast_cfg()
    fcn_cfg = FCNConfig(hidden_sizes=[32, 16], dropout_rate=0.0, output_size=1)
    model = FCNModel(fcn_cfg)
    loader = _fcn_loader()
    ckpt = tmp_path / "fcn.pt"
    Trainer(cfg, ckpt, tmp_path / "log.csv").train(model, loader, loader)

    fresh = FCNModel(fcn_cfg)
    ev = Evaluator(tmp_path)
    result = ev.evaluate(fresh, "fcn", ckpt, loader, loader, loader, 5, False)
    assert result.test_mse < float("inf")


def test_rnn_train_then_cold_evaluate(tmp_path):
    """Train RNN, cold-load checkpoint, evaluate — MSE must be finite."""
    cfg = _fast_cfg()
    model = RNNModel(RNNConfig(hidden_size=16, nonlinearity="tanh", num_layers=1, output_size=1))
    loader = _seq_loader()
    ckpt = tmp_path / "rnn.pt"
    Trainer(cfg, ckpt, tmp_path / "log.csv").train(model, loader, loader)

    fresh = RNNModel(RNNConfig(hidden_size=16, nonlinearity="tanh", num_layers=1, output_size=1))
    result = Evaluator(tmp_path).evaluate(fresh, "rnn", ckpt, loader, loader, loader, 5, False)
    assert result.test_mse < float("inf")


def test_lstm_train_then_cold_evaluate(tmp_path):
    """Train LSTM, cold-load checkpoint, evaluate — MSE must be finite."""
    cfg = _fast_cfg()
    model = LSTMModel(LSTMConfig(hidden_size=16, num_layers=1, dense_hidden_size=8, output_size=1))
    loader = _seq_loader()
    ckpt = tmp_path / "lstm.pt"
    Trainer(cfg, ckpt, tmp_path / "log.csv").train(model, loader, loader)

    fresh = LSTMModel(
        LSTMConfig(hidden_size=16, num_layers=1, dense_hidden_size=8, output_size=1)
    )
    result = Evaluator(tmp_path).evaluate(
        fresh, "lstm", ckpt, loader, loader, loader, 5, False
    )
    assert result.test_mse < float("inf")


# ── Gap 11: val MSE divergence → early stopping ───────────────────────────────

def test_val_mse_divergence_triggers_early_stopping(tmp_path):
    """When val loss never improves, early stopping fires before max_epochs."""
    cfg = _fast_cfg(learning_rate=0.0, max_epochs=50, early_stopping_patience=3)
    model = nn.Linear(15, 1)
    loader = _fcn_loader()
    result: TrainingResult = Trainer(cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        model, loader, loader
    )
    assert result.stopped_early
    assert result.epochs_trained < 50


def test_val_mse_divergence_epochs_capped_by_patience(tmp_path):
    """With patience=2 and no improvement, training stops at most at patience+1 epochs."""
    cfg = _fast_cfg(learning_rate=0.0, max_epochs=100, early_stopping_patience=2)
    model = nn.Linear(15, 1)
    loader = _fcn_loader()
    result = Trainer(cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        model, loader, loader
    )
    assert result.epochs_trained <= 3


def test_val_losses_do_not_decrease_when_lr_zero(tmp_path):
    """With lr=0, all val losses after epoch 1 should equal the first val loss."""
    cfg = _fast_cfg(learning_rate=0.0, max_epochs=5, early_stopping_patience=10)
    model = nn.Linear(15, 1)
    loader = _fcn_loader()
    result = Trainer(cfg, tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        model, loader, loader
    )
    first = result.val_losses[0]
    assert all(abs(v - first) < 1e-8 for v in result.val_losses)
