"""Integration tests for both the old signal_dataset pipeline and new neural_signal pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.models.fcn import FCNModel
from neural_signal.models.lstm import LSTMModel
from neural_signal.models.rnn import RNNModel
from neural_signal.services.data_loader import DataLoaderService
from neural_signal.services.evaluator import EvalResult, Evaluator
from neural_signal.services.preprocessor import Preprocessor
from neural_signal.services.trainer import Trainer
from neural_signal.services.visualizer import Visualizer
from neural_signal.shared.config import (
    FCNConfig,
    LSTMConfig,
    RNNConfig,
    TrainingConfig,
)


@pytest.fixture()
def fast_train_cfg() -> TrainingConfig:
    return TrainingConfig(
        batch_size=32, learning_rate=0.01, weight_decay=0.0,
        max_epochs=3, early_stopping_patience=10, val_split=0.2, random_seed=42,
    )


@pytest.fixture()
def fcn_cfg() -> FCNConfig:
    return FCNConfig(hidden_sizes=[32, 16], dropout_rate=0.0, output_size=1)


@pytest.fixture()
def rnn_cfg() -> RNNConfig:
    return RNNConfig(hidden_size=16, nonlinearity="tanh", num_layers=1, output_size=1)


@pytest.fixture()
def lstm_cfg() -> LSTMConfig:
    return LSTMConfig(hidden_size=16, num_layers=1, dense_hidden_size=8, output_size=1)


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


def test_full_fcn_pipeline(fast_train_cfg, fcn_cfg, tmp_path):
    model = FCNModel(fcn_cfg)
    loader = _fcn_loader()
    ckpt = tmp_path / "fcn.pt"
    result = Trainer(fast_train_cfg, ckpt, tmp_path / "log.csv").train(model, loader, loader)
    assert result.epochs_trained > 0 and ckpt.exists()

    ev = Evaluator(tmp_path)
    er = ev.evaluate(model, "fcn", ckpt, loader, loader, loader, result.epochs_trained, False)
    assert er.test_mse < float("inf")

    df = ev.save_comparison_table([er], tmp_path / "table.csv")
    assert (tmp_path / "table.csv").exists() and len(df) == 1


def test_full_rnn_pipeline(fast_train_cfg, rnn_cfg, tmp_path):
    model = RNNModel(rnn_cfg)
    loader = _seq_loader()
    ckpt = tmp_path / "rnn.pt"
    result = Trainer(fast_train_cfg, ckpt, tmp_path / "log.csv").train(model, loader, loader)
    assert ckpt.exists() and result.epochs_trained > 0


def test_full_lstm_pipeline(fast_train_cfg, lstm_cfg, tmp_path):
    model = LSTMModel(lstm_cfg)
    loader = _seq_loader()
    ckpt = tmp_path / "lstm.pt"
    result = Trainer(fast_train_cfg, ckpt, tmp_path / "log.csv").train(model, loader, loader)
    assert ckpt.exists() and result.epochs_trained > 0


def test_data_loader_to_preprocessor_pipeline(tiny_npz, tmp_path):
    bundle = DataLoaderService(tiny_npz, 32).load()
    pre = Preprocessor(tmp_path / "scaler.json")
    x_norm = pre.fit_transform(bundle.X_train)
    assert x_norm.shape == bundle.X_train.shape
    pre.save_params()
    assert (tmp_path / "scaler.json").exists()


def test_visualizer_produces_required_plots(tmp_path):
    viz = Visualizer(tmp_path)
    y = np.sin(np.linspace(0, 4 * np.pi, 300)).astype(np.float32)
    noisy = (y + 0.05).astype(np.float32)
    preds = {"fcn": y + 0.01, "rnn": y + 0.02, "lstm": y + 0.005}
    results = [
        EvalResult("fcn", 0.05, 0.06, 0.07, 10, False),
        EvalResult("rnn", 0.04, 0.05, 0.06, 12, False),
        EvalResult("lstm", 0.03, 0.04, 0.05, 15, True),
    ]
    viz.plot_clean_noisy_predicted(y, noisy, preds)
    for name in ["fcn", "rnn", "lstm"]:
        viz.plot_loss_curves(name, [0.5, 0.4], [0.6, 0.5])
    viz.plot_mse_comparison(results)
    for name in ["fcn", "rnn", "lstm"]:
        viz.plot_residuals(name, y, y + 0.01)
    viz.plot_pred_vs_actual({"fcn": (y, y + 0.01), "rnn": (y, y), "lstm": (y, y - 0.01)})

    assert (tmp_path / "clean_noisy_predicted.png").exists()
    for name in ["fcn", "rnn", "lstm"]:
        assert (tmp_path / f"loss_curves_{name}.png").exists()
    assert (tmp_path / "mse_comparison.png").exists()
