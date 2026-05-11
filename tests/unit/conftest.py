"""Unit-test-level fixtures for signal_dataset legacy tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signal_dataset.constants import TWO_PI


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
