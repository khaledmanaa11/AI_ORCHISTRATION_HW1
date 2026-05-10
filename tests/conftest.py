"""Shared pytest fixtures for all unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from signal_dataset.constants import TWO_PI

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


# ───────────────────────────── signal_dataset legacy fixtures ─────────────────

@pytest.fixture()
def sample_config_dict() -> dict:
    return {
        "dataset": {
            "duration_sec": 10,
            "sample_rate_hz": 1000,
            "window_size": 10,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
        },
        "signals": [
            {"id": "s1", "amplitude": 2.0, "frequency_hz": 5, "phase_rad": 0},
            {"id": "s2", "amplitude": 1.5, "frequency_hz": 15, "phase_rad": 0.7854},
            {"id": "s3", "amplitude": 0.8, "frequency_hz": 50, "phase_rad": 0},
            {"id": "s4", "amplitude": 0.3, "frequency_hz": 100, "phase_rad": 0},
        ],
        "noise": {
            "gaussian": {
                "sigma_amp": [0.05, 0.05, 0.03, 0.02],
                "sigma_phase": [0.05, 0.05, 0.03, 0.02],
            },
            "burst": {
                "probability": 0.01,
                "duration_range_samples": [5, 20],
                "amp_magnitude": 0.5,
                "phase_magnitude": 0.3,
            },
        },
        "output": {
            "dataset_path": "data/dataset.npz",
            "raw_signals_path": "data/signals_raw.npz",
            "results_dir": "results/",
        },
    }


@pytest.fixture()
def time_axis_fixture() -> np.ndarray:
    return np.linspace(0, 10, 10000, endpoint=False)


@pytest.fixture()
def small_time_axis() -> np.ndarray:
    return np.linspace(0, 0.1, 100, endpoint=False)


@pytest.fixture()
def clean_s1_fixture(small_time_axis: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(TWO_PI * 5 * small_time_axis)


@pytest.fixture()
def clean_s2_fixture(small_time_axis: np.ndarray) -> np.ndarray:
    return 1.5 * np.sin(TWO_PI * 15 * small_time_axis + 0.7854)


@pytest.fixture()
def clean_s3_fixture(small_time_axis: np.ndarray) -> np.ndarray:
    return 0.8 * np.sin(TWO_PI * 50 * small_time_axis)


@pytest.fixture()
def clean_s4_fixture(small_time_axis: np.ndarray) -> np.ndarray:
    return 0.3 * np.sin(TWO_PI * 100 * small_time_axis)


@pytest.fixture()
def clean_s5_fixture(
    clean_s1_fixture, clean_s2_fixture, clean_s3_fixture, clean_s4_fixture
) -> np.ndarray:
    return clean_s1_fixture + clean_s2_fixture + clean_s3_fixture + clean_s4_fixture


@pytest.fixture()
def noisy_signals_fixture(
    small_time_axis, clean_s1_fixture, clean_s2_fixture, clean_s3_fixture, clean_s4_fixture
) -> dict:
    rng = np.random.default_rng(42)
    n = len(small_time_axis)
    noisy: dict = {}
    for sid, clean in zip(
        ["s1", "s2", "s3", "s4"],
        [clean_s1_fixture, clean_s2_fixture, clean_s3_fixture, clean_s4_fixture],
        strict=True,
    ):
        noisy[sid] = clean + rng.normal(0, 0.01, n)
    noisy["s5"] = sum(noisy[s] for s in ["s1", "s2", "s3", "s4"])
    return noisy


@pytest.fixture()
def tmp_data_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture()
def window_size_fixture() -> int:
    return 10


@pytest.fixture()
def random_seed_fixture() -> int:
    return 42
