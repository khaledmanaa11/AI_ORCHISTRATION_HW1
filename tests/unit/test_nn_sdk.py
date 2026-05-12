"""Tests for sdk/sdk.py — NeuralSignalSDK orchestration."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.sdk.sdk import NeuralSignalSDK
from neural_signal.services.data_loader import DataBundle
from neural_signal.services.evaluator import EvalResult
from neural_signal.services.trainer import TrainingResult


def _loader(n: int = 60, f: int = 15) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.randn(n, f), torch.randn(n, 1)), batch_size=16, shuffle=False
    )


def _sloader(n: int = 60) -> DataLoader:
    # Shape (N, seq_len=10, features=6): window(1) + C broadcast(5)
    return DataLoader(
        TensorDataset(torch.randn(n, 10, 6), torch.randn(n, 1)), batch_size=16, shuffle=False
    )


def _make_bundle() -> DataBundle:
    return DataBundle(
        train_loader=_loader(), val_loader=_loader(20), test_loader=_loader(20),
        seq_train_loader=_sloader(), seq_val_loader=_sloader(20), seq_test_loader=_sloader(20),
        X_train=np.zeros((60, 10), dtype="f"), X_val=np.zeros((20, 10), dtype="f"),
        X_test=np.zeros((20, 10), dtype="f"), C_train=np.zeros((60, 5), dtype="f"),
        C_val=np.zeros((20, 5), dtype="f"), C_test=np.zeros((20, 5), dtype="f"),
        y_train=np.random.randn(60, 1).astype("f"),
        y_val=np.zeros((20, 1), dtype="f"), y_test=np.zeros((20, 1), dtype="f"),
    )


@pytest.fixture()
def sdk(config_path, rate_limits_path) -> NeuralSignalSDK:
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    sdk._cfg.training.__dict__["max_epochs"] = 2
    sdk._cfg.training.__dict__["early_stopping_patience"] = 10
    return sdk


def test_sdk_instantiates(sdk):
    assert sdk is not None


def test_sdk_has_cfg(sdk):
    assert sdk._cfg is not None


def test_sdk_cfg_has_training(sdk):
    assert sdk._cfg.training is not None


def test_sdk_cfg_batch_size_64(sdk):
    assert sdk._cfg.training.batch_size == 64


def test_sdk_train_model_fcn(sdk, tmp_path):
    bundle = _make_bundle()
    sdk._cfg.output.__dict__["fcn_checkpoint"] = str(tmp_path / "fcn.pt")
    sdk._cfg.output.__dict__["results_dir"] = str(tmp_path) + "/"
    with patch.object(sdk, "_load_data", return_value=bundle):
        result = sdk.train_model("fcn")
    assert isinstance(result, TrainingResult)
    assert result.epochs_trained > 0


def test_sdk_train_model_rnn(sdk, tmp_path):
    bundle = _make_bundle()
    sdk._cfg.output.__dict__["rnn_checkpoint"] = str(tmp_path / "rnn.pt")
    sdk._cfg.output.__dict__["results_dir"] = str(tmp_path) + "/"
    with patch.object(sdk, "_load_data", return_value=bundle):
        result = sdk.train_model("rnn")
    assert isinstance(result, TrainingResult)


def test_sdk_train_model_lstm(sdk, tmp_path):
    bundle = _make_bundle()
    sdk._cfg.output.__dict__["lstm_checkpoint"] = str(tmp_path / "lstm.pt")
    sdk._cfg.output.__dict__["results_dir"] = str(tmp_path) + "/"
    with patch.object(sdk, "_load_data", return_value=bundle):
        result = sdk.train_model("lstm")
    assert isinstance(result, TrainingResult)


def test_sdk_evaluate_all(sdk, tmp_path):
    bundle = _make_bundle()
    sdk._cfg.output.__dict__["results_dir"] = str(tmp_path) + "/"
    sdk._cfg.output.__dict__["comparison_table_path"] = str(tmp_path / "table.csv")
    sdk._cfg.output.__dict__["fcn_checkpoint"] = str(tmp_path / "fcn.pt")
    sdk._cfg.output.__dict__["rnn_checkpoint"] = str(tmp_path / "rnn.pt")
    sdk._cfg.output.__dict__["lstm_checkpoint"] = str(tmp_path / "lstm.pt")
    sdk._cfg.output.__dict__["scaler_params_path"] = str(tmp_path / "sc.json")
    sdk._cfg.output.__dict__["training_log_fcn"] = str(tmp_path / "log_fcn.csv")
    sdk._cfg.output.__dict__["training_log_rnn"] = str(tmp_path / "log_rnn.csv")
    sdk._cfg.output.__dict__["training_log_lstm"] = str(tmp_path / "log_lstm.csv")
    with patch.object(sdk, "_load_data", return_value=bundle):
        for name in ("fcn", "rnn", "lstm"):
            sdk.train_model(name)
        results = sdk.evaluate_all()
    assert len(results) == 3
    assert all(isinstance(v, EvalResult) for v in results.values())


def _patch_output(sdk, tmp_path) -> None:
    sdk._cfg.output.__dict__["results_dir"] = str(tmp_path) + "/"
    sdk._cfg.output.__dict__["comparison_table_path"] = str(tmp_path / "table.csv")
    sdk._cfg.output.__dict__["fcn_checkpoint"] = str(tmp_path / "fcn.pt")
    sdk._cfg.output.__dict__["rnn_checkpoint"] = str(tmp_path / "rnn.pt")
    sdk._cfg.output.__dict__["lstm_checkpoint"] = str(tmp_path / "lstm.pt")
    sdk._cfg.output.__dict__["scaler_params_path"] = str(tmp_path / "scaler.json")

def test_sdk_run_all(sdk, tmp_path):
    from neural_signal.sdk.sdk import RunResult
    bundle = _make_bundle()
    _patch_output(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=bundle):
        result = sdk.run_all()
    assert isinstance(result, RunResult)
    assert (tmp_path / "table.csv").exists()


def test_sdk_run_all_returns_run_result(sdk, tmp_path):
    from neural_signal.sdk.sdk import RunResult
    bundle = _make_bundle()
    _patch_output(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=bundle):
        result = sdk.run_all()
    assert isinstance(result, RunResult)


def test_run_result_has_comparison_df(sdk, tmp_path):
    import pandas as pd
    bundle = _make_bundle()
    _patch_output(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=bundle):
        result = sdk.run_all()
    assert isinstance(result.comparison_df, pd.DataFrame)


def test_sdk_raises_on_unknown_model_name(sdk):
    with pytest.raises(ValueError, match="Unknown model"):
        sdk.train_model("invalid_model")


def test_sdk_get_version_returns_string(sdk):
    version = sdk.get_version()
    assert isinstance(version, str) and len(version) > 0
