"""Shared pytest fixtures for unit and integration tests."""

import numpy as np
import pytest

from signal_dataset.constants import TWO_PI


@pytest.fixture
def config_path():
    """Path to the real setup.json config file."""
    from pathlib import Path
    return Path(__file__).parent.parent / "config" / "setup.json"


@pytest.fixture
def rate_limits_path():
    """Path to the real rate_limits.json config file."""
    from pathlib import Path
    return Path(__file__).parent.parent / "config" / "rate_limits.json"


@pytest.fixture
def sample_config_dict():
    """Minimal valid config dict — no file I/O."""
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
            "gaussian": {"sigma_amp": [0.05, 0.05, 0.03, 0.02], "sigma_phase": [0.05, 0.05, 0.03, 0.02]},
            "burst": {"probability": 0.01, "duration_range_samples": [5, 20], "amp_magnitude": 0.5, "phase_magnitude": 0.3},
        },
        "output": {"dataset_path": "data/dataset.npz", "raw_signals_path": "data/signals_raw.npz", "results_dir": "results/"},
    }


@pytest.fixture
def time_axis_fixture():
    """Full 10-second time axis at 1000 Hz — 10000 samples."""
    return np.linspace(0, 10, 10000, endpoint=False)


@pytest.fixture
def small_time_axis():
    """Small 100-sample time axis for fast tests."""
    return np.linspace(0, 0.1, 100, endpoint=False)


@pytest.fixture
def clean_s1_fixture(small_time_axis):
    """Precomputed s1 on the small axis."""
    return 2.0 * np.sin(TWO_PI * 5 * small_time_axis)


@pytest.fixture
def clean_s2_fixture(small_time_axis):
    """Precomputed s2 on the small axis."""
    return 1.5 * np.sin(TWO_PI * 15 * small_time_axis + 0.7854)


@pytest.fixture
def clean_s3_fixture(small_time_axis):
    """Precomputed s3 on the small axis."""
    return 0.8 * np.sin(TWO_PI * 50 * small_time_axis)


@pytest.fixture
def clean_s4_fixture(small_time_axis):
    """Precomputed s4 on the small axis."""
    return 0.3 * np.sin(TWO_PI * 100 * small_time_axis)


@pytest.fixture
def clean_s5_fixture(clean_s1_fixture, clean_s2_fixture, clean_s3_fixture, clean_s4_fixture):
    """Composite clean signal s5 = s1+s2+s3+s4 on the small axis."""
    return clean_s1_fixture + clean_s2_fixture + clean_s3_fixture + clean_s4_fixture


@pytest.fixture
def noisy_signals_fixture(
    small_time_axis, clean_s1_fixture, clean_s2_fixture, clean_s3_fixture, clean_s4_fixture
):
    """Dict of 5 noisy arrays on the small axis (tiny noise for test stability)."""
    rng = np.random.default_rng(42)
    n = len(small_time_axis)
    noisy = {}
    for sid, clean in zip(
        ["s1", "s2", "s3", "s4"],
        [clean_s1_fixture, clean_s2_fixture, clean_s3_fixture, clean_s4_fixture],
        strict=True,
    ):
        n_amp = rng.normal(0, 0.01, n)
        noisy[sid] = clean + n_amp  # simplified for fixture
    noisy["s5"] = sum(noisy[s] for s in ["s1", "s2", "s3", "s4"])
    return noisy


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Isolated temporary directory for file I/O tests."""
    return tmp_path


@pytest.fixture
def window_size_fixture():
    """Default window size used across tests."""
    return 10


@pytest.fixture
def random_seed_fixture():
    """Default random seed used across tests."""
    return 42
