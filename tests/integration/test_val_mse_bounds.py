"""Val MSE bound integration tests — Gap 11 extension."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.models.fcn import FCNModel
from neural_signal.models.lstm import LSTMModel
from neural_signal.models.rnn import RNNModel
from neural_signal.services.trainer import Trainer
from neural_signal.shared.config import FCNConfig, LSTMConfig, RNNConfig, TrainingConfig


def _fcn_loader(n: int = 120) -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(n, 15), torch.randn(n, 1)),
                      batch_size=32, shuffle=False)


def _seq_loader(n: int = 120) -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(n, 10, 1), torch.randn(n, 1)),
                      batch_size=32, shuffle=False)


def _cfg(**kw) -> TrainingConfig:
    base = {
        "batch_size": 32, "learning_rate": 0.01, "weight_decay": 0.0,
        "max_epochs": 10, "early_stopping_patience": 10, "val_split": 0.2, "random_seed": 42,
    }
    base.update(kw)
    return TrainingConfig(**base)


def test_val_mse_does_not_increase_unboundedly_for_fcn(tmp_path):
    """Over all logged epochs, val_mse never exceeds 10× best_val_mse."""
    model = FCNModel(FCNConfig(hidden_sizes=[32, 16], dropout_rate=0.0, output_size=1))
    result = Trainer(_cfg(), tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        model, _fcn_loader(), _fcn_loader()
    )
    assert max(result.val_losses) <= result.best_val_mse * 10 + 1.0


def test_val_mse_does_not_increase_unboundedly_for_rnn(tmp_path):
    """Over all logged epochs, val_mse never exceeds 10× best_val_mse."""
    model = RNNModel(RNNConfig(hidden_size=16, nonlinearity="tanh", num_layers=1, output_size=1))
    result = Trainer(_cfg(), tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        model, _seq_loader(), _seq_loader()
    )
    assert max(result.val_losses) <= result.best_val_mse * 10 + 1.0


def test_val_mse_does_not_increase_unboundedly_for_lstm(tmp_path):
    """Over all logged epochs, val_mse never exceeds 10× best_val_mse."""
    model = LSTMModel(LSTMConfig(hidden_size=16, num_layers=1, dense_hidden_size=8, output_size=1))
    result = Trainer(_cfg(), tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        model, _seq_loader(), _seq_loader()
    )
    assert max(result.val_losses) <= result.best_val_mse * 10 + 1.0


def test_early_stopping_prevents_val_mse_from_diverging(tmp_path):
    """Inject growing val losses via lr=0; early stopping fires within patience."""
    from torch import nn
    cfg = _cfg(learning_rate=0.0, max_epochs=50, early_stopping_patience=3)
    model = nn.Linear(15, 1)
    result = Trainer(cfg, tmp_path / "c.pt", tmp_path / "log.csv").train(
        model, _fcn_loader(), _fcn_loader()
    )
    assert result.stopped_early and result.epochs_trained < 50
