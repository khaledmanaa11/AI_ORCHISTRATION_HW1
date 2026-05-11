"""Shared pytest fixtures for all unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

# ───────────────────────────── paths ──────────────────────────────────────────

@pytest.fixture()
def config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "setup.json"


@pytest.fixture()
def rate_limits_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "rate_limits.json"


# ───────────────────────────── neural_signal fixtures ─────────────────────────

@pytest.fixture()
def mock_training_config() -> dict:
    return {
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 5,
        "early_stopping_patience": 3,
        "val_split": 0.2,
        "random_seed": 42,
    }


@pytest.fixture()
def mock_fcn_config() -> dict:
    return {"hidden_sizes": [128, 64], "dropout_rate": 0.1, "output_size": 1}


@pytest.fixture()
def mock_rnn_config() -> dict:
    return {"hidden_size": 64, "nonlinearity": "tanh", "num_layers": 1, "output_size": 1}


@pytest.fixture()
def mock_lstm_config() -> dict:
    return {"hidden_size": 64, "num_layers": 1, "dense_hidden_size": 32, "output_size": 1}


@pytest.fixture()
def tiny_x_train() -> torch.Tensor:
    return torch.randn(100, 10)


@pytest.fixture()
def tiny_c_train() -> torch.Tensor:
    t = torch.zeros(100, 5)
    t[:, 1] = 1.0
    return t


@pytest.fixture()
def tiny_y_train() -> torch.Tensor:
    return torch.randn(100, 1)


@pytest.fixture()
def tiny_x_input_fcn(tiny_x_train, tiny_c_train) -> torch.Tensor:
    return torch.cat([tiny_x_train, tiny_c_train], dim=1)


@pytest.fixture()
def tiny_x_seq_rnn(tiny_x_train) -> torch.Tensor:
    return tiny_x_train.unsqueeze(-1)


@pytest.fixture()
def tmp_results_dir(tmp_path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture()
def tmp_checkpoint_path(tmp_path) -> Path:
    return tmp_path / "model_best.pt"


@pytest.fixture()
def batch_size_fixture() -> int:
    return 16


@pytest.fixture()
def tiny_npz(tmp_path) -> Path:
    """Small synthetic dataset.npz with all required keys."""
    rng = np.random.default_rng(0)
    n_tr, n_va, n_te = 200, 50, 50
    path = tmp_path / "dataset.npz"
    np.savez(
        path,
        X_train=rng.random((n_tr, 10)).astype(np.float32),
        X_val=rng.random((n_va, 10)).astype(np.float32),
        X_test=rng.random((n_te, 10)).astype(np.float32),
        C_train=np.eye(5, dtype=np.float32)[rng.integers(0, 5, n_tr)],
        C_val=np.eye(5, dtype=np.float32)[rng.integers(0, 5, n_va)],
        C_test=np.eye(5, dtype=np.float32)[rng.integers(0, 5, n_te)],
        y_train=rng.random((n_tr, 1)).astype(np.float32),
        y_val=rng.random((n_va, 1)).astype(np.float32),
        y_test=rng.random((n_te, 1)).astype(np.float32),
    )
    return path
