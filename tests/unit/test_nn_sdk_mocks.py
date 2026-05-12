"""SDK mock tests — verify orchestration calls to service layer."""

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
    return DataLoader(TensorDataset(torch.randn(n, f), torch.randn(n, 1)),
                      batch_size=16, shuffle=False)


def _sloader(n: int = 60) -> DataLoader:
    # Shape (N, seq_len=10, features=6): window(1) + C broadcast(5)
    return DataLoader(TensorDataset(torch.randn(n, 10, 6), torch.randn(n, 1)),
                      batch_size=16, shuffle=False)


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


def _patch(sdk, tmp_path) -> None:
    sdk._cfg.training.__dict__["max_epochs"] = 1
    sdk._cfg.output.__dict__["results_dir"] = str(tmp_path) + "/"
    sdk._cfg.output.__dict__["comparison_table_path"] = str(tmp_path / "t.csv")
    sdk._cfg.output.__dict__["fcn_checkpoint"] = str(tmp_path / "fcn.pt")
    sdk._cfg.output.__dict__["rnn_checkpoint"] = str(tmp_path / "rnn.pt")
    sdk._cfg.output.__dict__["lstm_checkpoint"] = str(tmp_path / "lstm.pt")
    sdk._cfg.output.__dict__["scaler_params_path"] = str(tmp_path / "sc.json")
    sdk._cfg.output.__dict__["training_log_fcn"] = str(tmp_path / "log_fcn.csv")
    sdk._cfg.output.__dict__["training_log_rnn"] = str(tmp_path / "log_rnn.csv")
    sdk._cfg.output.__dict__["training_log_lstm"] = str(tmp_path / "log_lstm.csv")


def test_sdk_calls_data_loader_service(config_path, rate_limits_path, tmp_path):
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()) as mock_load:
        sdk.train_model("fcn")
    mock_load.assert_called_once()


def test_sdk_calls_trainer_for_each_model(config_path, rate_limits_path, tmp_path):
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()):
        result = sdk.train_model("fcn")
    assert isinstance(result, TrainingResult)


def test_sdk_calls_evaluator(config_path, rate_limits_path, tmp_path):
    """evaluate_all() requires trained checkpoints; train all models first."""
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()):
        for name in ("fcn", "rnn", "lstm"):
            sdk.train_model(name)
        results = sdk.evaluate_all()
    assert all(isinstance(v, EvalResult) for v in results.values())


def test_sdk_passes_correct_log_path_for_fcn(config_path, rate_limits_path, tmp_path):
    """SDK uses configured training_log_fcn path — log file should be created."""
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()):
        sdk.train_model("fcn")
    assert (tmp_path / "log_fcn.csv").exists()


def test_sdk_passes_correct_log_path_for_rnn(config_path, rate_limits_path, tmp_path):
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()):
        sdk.train_model("rnn")
    assert (tmp_path / "log_rnn.csv").exists()


def test_sdk_passes_correct_log_path_for_lstm(config_path, rate_limits_path, tmp_path):
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()):
        sdk.train_model("lstm")
    assert (tmp_path / "log_lstm.csv").exists()


def test_sdk_calls_preprocessor(config_path, rate_limits_path, tmp_path):
    """Preprocessor is wired via OutputConfig.scaler_params_path — path is correct."""
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    assert sdk._cfg.output.scaler_params_path == str(tmp_path / "sc.json")


def test_sdk_calls_visualizer(config_path, rate_limits_path, tmp_path):
    """After evaluate_all, at least one PNG should exist in results dir."""
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    sdk._cfg.training.__dict__["max_epochs"] = 1
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()):
        sdk.run_all()
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) > 0


def test_sdk_evaluate_all_works_without_in_memory_models(config_path, rate_limits_path, tmp_path):
    """evaluate_all() loads from checkpoint files, not cached model objects."""
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()):
        for name in ("fcn", "rnn", "lstm"):
            sdk.train_model(name)
        sdk._train_results.clear()
        results = sdk.evaluate_all()
    assert "fcn" in results


def test_evaluate_all_raises_if_checkpoint_missing(config_path, rate_limits_path, tmp_path):
    """evaluate_all() raises FileNotFoundError when no checkpoint has been saved."""
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()), pytest.raises(FileNotFoundError):
        sdk.evaluate_all()


def test_evaluate_all_loads_correct_checkpoint_per_model(config_path, rate_limits_path, tmp_path):
    """Each model's evaluate step uses its own .pt file."""
    sdk = NeuralSignalSDK(config_path, rate_limits_path)
    _patch(sdk, tmp_path)
    with patch.object(sdk, "_load_data", return_value=_make_bundle()):
        for name in ("fcn", "rnn", "lstm"):
            sdk.train_model(name)
        results = sdk.evaluate_all()
    assert set(results.keys()) == {"fcn", "rnn", "lstm"}
    assert (tmp_path / "fcn.pt").exists()
    assert (tmp_path / "rnn.pt").exists()
    assert (tmp_path / "lstm.pt").exists()
