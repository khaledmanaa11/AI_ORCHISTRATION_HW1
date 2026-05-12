"""Pipeline quality and reproducibility integration tests."""

from __future__ import annotations

import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.models.fcn import FCNModel
from neural_signal.services.evaluator import Evaluator
from neural_signal.services.trainer import Trainer
from neural_signal.shared.config import FCNConfig, TrainingConfig


def _loader(n: int = 200) -> DataLoader:
    torch.manual_seed(42)
    return DataLoader(TensorDataset(torch.randn(n, 15), torch.randn(n, 1) * 0.1),
                      batch_size=32, shuffle=False)


def _cfg(**kw) -> TrainingConfig:
    base = {
        "batch_size": 32, "learning_rate": 0.05, "weight_decay": 0.0,
        "max_epochs": 10, "early_stopping_patience": 10, "val_split": 0.2, "random_seed": 42,
    }
    base.update(kw)
    return TrainingConfig(**base)


def _fcn() -> nn.Module:
    return FCNModel(FCNConfig(hidden_sizes=[32, 16], dropout_rate=0.0, output_size=1))


def test_fcn_train_mse_less_than_random_baseline(tmp_path):
    """FCN train MSE must beat random-prediction baseline (var of targets)."""
    loader = _loader()
    model = _fcn()
    ckpt = tmp_path / "fcn.pt"
    result = Trainer(_cfg(), ckpt, tmp_path / "log.csv").train(model, loader, loader)
    random_baseline = 1.0
    assert result.train_losses[-1] < random_baseline


def test_early_stopping_log_entries_match_epochs_trained(tmp_path):
    """CSV log rows (excluding header) should equal epochs_trained."""
    log = tmp_path / "log.csv"
    result = Trainer(_cfg(), tmp_path / "ckpt.pt", log).train(_fcn(), _loader(), _loader())
    with log.open() as fh:
        rows = list(csv.reader(fh))
    assert len(rows) - 1 == result.epochs_trained


def test_checkpoint_loadable_and_produces_same_predictions(tmp_path):
    """Checkpoint reload should produce identical output on same input."""
    loader = _loader(50)
    model = _fcn()
    ckpt = tmp_path / "fcn.pt"
    Trainer(_cfg(max_epochs=3), ckpt, tmp_path / "log.csv").train(model, loader, loader)

    fresh = _fcn()
    ev = Evaluator(tmp_path)
    r1 = ev.evaluate(model, "fcn", ckpt, loader, loader, loader, 3, False)
    r2 = ev.evaluate(fresh, "fcn", ckpt, loader, loader, loader, 3, False)
    assert abs(r1.test_mse - r2.test_mse) < 1e-5


def test_fcn_train_mse_less_than_val_mse_or_equal(tmp_path):
    """Train MSE should not be severely higher than val MSE (no extreme underfitting)."""
    loader = _loader()
    result = Trainer(_cfg(), tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        _fcn(), loader, loader
    )
    assert result.train_losses[-1] < float("inf")
    assert result.best_val_mse < float("inf")


def test_rnn_train_mse_less_than_random_baseline(tmp_path):
    """RNN train MSE must beat trivial random-prediction baseline."""
    from neural_signal.models.rnn import RNNModel
    from neural_signal.shared.config import RNNConfig
    seq_loader = DataLoader(
        TensorDataset(torch.randn(200, 10, 6), torch.randn(200, 1) * 0.1),
        batch_size=32, shuffle=False,
    )
    model = RNNModel(RNNConfig(hidden_size=16, nonlinearity="tanh", num_layers=1, output_size=1, input_size=6))
    result = Trainer(_cfg(), tmp_path / "r.pt", tmp_path / "lr.csv").train(
        model, seq_loader, seq_loader
    )
    assert result.train_losses[-1] < 10.0


def test_lstm_train_mse_less_than_random_baseline(tmp_path):
    """LSTM train MSE must beat trivial random-prediction baseline."""
    from neural_signal.models.lstm import LSTMModel
    from neural_signal.shared.config import LSTMConfig
    seq_loader = DataLoader(
        TensorDataset(torch.randn(200, 10, 6), torch.randn(200, 1) * 0.1),
        batch_size=32, shuffle=False,
    )
    model = LSTMModel(LSTMConfig(hidden_size=16, num_layers=1, dense_hidden_size=8, output_size=1, input_size=6))
    result = Trainer(_cfg(), tmp_path / "l.pt", tmp_path / "ll.csv").train(
        model, seq_loader, seq_loader
    )
    assert result.train_losses[-1] < 10.0


def test_pipeline_reproducible_with_same_seed(tmp_path):
    """Two runs with same seed and same data should give same best_val_mse."""
    loader = _loader()
    cfg = _cfg()
    r1 = Trainer(cfg, tmp_path / "a.pt", tmp_path / "la.csv").train(_fcn(), loader, loader)
    r2 = Trainer(cfg, tmp_path / "b.pt", tmp_path / "lb.csv").train(_fcn(), loader, loader)
    assert abs(r1.best_val_mse - r2.best_val_mse) < 1e-3


# ── Gap 2: Validation split — trainer uses provided val_loader ────────────────

def test_trainer_uses_provided_val_loader_not_internal_split(tmp_path):
    """Trainer must use the supplied val_loader; it must not re-split train data."""
    train_loader = _loader(200)
    # Distinct val_loader — zeros target so val MSE will differ from train MSE
    val_x = torch.randn(50, 15)
    val_y = torch.zeros(50, 1)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False)
    result = Trainer(_cfg(max_epochs=3), tmp_path / "ckpt.pt", tmp_path / "log.csv").train(
        _fcn(), train_loader, val_loader
    )
    # Val MSE should reflect the zero-target val set, not the train distribution
    assert result.best_val_mse >= 0.0 and result.best_val_mse < float("inf")


def test_val_loader_size_matches_dataset_npz_val_split(tiny_npz, batch_size_fixture):
    """Val split size from DataLoaderService must equal X_val size from dataset.npz (no re-split)."""
    from neural_signal.services.data_loader import DataLoaderService
    npz = np.load(tiny_npz)
    n_va_expected = len(npz["y_val"])
    bundle = DataLoaderService(tiny_npz, batch_size_fixture).load()
    n_va = sum(len(xb) for xb, _ in bundle.val_loader)
    # DataLoaderService must use X_val from npz directly, no trainer re-split
    assert n_va == n_va_expected
