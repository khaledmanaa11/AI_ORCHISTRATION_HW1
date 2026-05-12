"""End-to-end NeuralSignalSDK.run_all() integration tests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.sdk._types import RunResult
from neural_signal.sdk.sdk import NeuralSignalSDK
from neural_signal.services.data_loader import DataBundle


def _loader(n: int = 80, f: int = 15) -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(n, f), torch.randn(n, 1)),
                      batch_size=32, shuffle=False)


def _sloader(n: int = 80) -> DataLoader:
    # Shape (N, seq_len=10, features=6): window(1) + C broadcast(5)
    return DataLoader(TensorDataset(torch.randn(n, 10, 6), torch.randn(n, 1)),
                      batch_size=32, shuffle=False)


def _bundle() -> DataBundle:
    return DataBundle(
        train_loader=_loader(), val_loader=_loader(20), test_loader=_loader(20),
        seq_train_loader=_sloader(), seq_val_loader=_sloader(20), seq_test_loader=_sloader(20),
        X_train=np.zeros((80, 10), dtype="f"), X_val=np.zeros((20, 10), dtype="f"),
        X_test=np.zeros((20, 10), dtype="f"), C_train=np.zeros((80, 5), dtype="f"),
        C_val=np.zeros((20, 5), dtype="f"), C_test=np.zeros((20, 5), dtype="f"),
        y_train=np.random.randn(80, 1).astype("f"),
        y_val=np.zeros((20, 1), dtype="f"), y_test=np.zeros((20, 1), dtype="f"),
    )


def _make_sdk(config_path, rate_limits_path, tmp_path) -> NeuralSignalSDK:
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    sdk._cfg.training.__dict__["max_epochs"] = 1
    sdk._cfg.output.__dict__["results_dir"] = str(tmp_path) + "/"
    sdk._cfg.output.__dict__["comparison_table_path"] = str(tmp_path / "table.csv")
    sdk._cfg.output.__dict__["fcn_checkpoint"] = str(tmp_path / "fcn.pt")
    sdk._cfg.output.__dict__["rnn_checkpoint"] = str(tmp_path / "rnn.pt")
    sdk._cfg.output.__dict__["lstm_checkpoint"] = str(tmp_path / "lstm.pt")
    sdk._cfg.output.__dict__["scaler_params_path"] = str(tmp_path / "scaler.json")
    sdk._cfg.output.__dict__["training_log_fcn"] = str(tmp_path / "log_fcn.csv")
    sdk._cfg.output.__dict__["training_log_rnn"] = str(tmp_path / "log_rnn.csv")
    sdk._cfg.output.__dict__["training_log_lstm"] = str(tmp_path / "log_lstm.csv")
    return sdk


def test_full_pipeline_runs_without_error(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        result = sdk.run_all()
    assert isinstance(result, RunResult)


def test_run_all_creates_comparison_table_csv(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    assert (tmp_path / "table.csv").exists()


def test_run_all_creates_fcn_checkpoint(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    assert (tmp_path / "fcn.pt").exists()


def test_run_all_creates_rnn_checkpoint(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    assert (tmp_path / "rnn.pt").exists()


def test_run_all_creates_lstm_checkpoint(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    assert (tmp_path / "lstm.pt").exists()


def test_comparison_table_has_three_rows(config_path, rate_limits_path, tmp_path):
    import pandas as pd
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    df = pd.read_csv(tmp_path / "table.csv")
    assert len(df) == 3


def test_all_mse_values_are_finite(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        result = sdk.run_all()
    for r in (result.fcn_eval, result.rnn_eval, result.lstm_eval):
        assert r.test_mse < float("inf")
        assert r.val_mse < float("inf")


def test_all_mse_values_are_non_negative(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        result = sdk.run_all()
    for r in (result.fcn_eval, result.rnn_eval, result.lstm_eval):
        assert r.test_mse >= 0


def test_run_all_creates_clean_noisy_predicted_png(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    assert (tmp_path / "clean_noisy_predicted.png").exists()


def test_run_all_creates_loss_curves_fcn_png(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    assert (tmp_path / "loss_curves_fcn.png").exists()


def test_run_all_creates_loss_curves_rnn_png(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    assert (tmp_path / "loss_curves_rnn.png").exists()


def test_run_all_creates_loss_curves_lstm_png(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    assert (tmp_path / "loss_curves_lstm.png").exists()


def test_run_all_creates_mse_comparison_png(config_path, rate_limits_path, tmp_path):
    sdk = _make_sdk(config_path, rate_limits_path, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_bundle()):
        sdk.run_all()
    assert (tmp_path / "mse_comparison.png").exists()
