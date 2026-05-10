"""Unit tests for services/dataset_builder.py — DatasetBuilder."""

import numpy as np
import pytest

from signal_dataset.services.dataset_builder import DatasetBuilder
from signal_dataset.shared.config import DatasetConfig, OutputConfig

W = 10
N = 1000  # total windows


@pytest.fixture
def dataset_cfg():
    return DatasetConfig(
        duration_sec=10,
        sample_rate_hz=1000,
        window_size=W,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )


@pytest.fixture
def output_cfg(tmp_path):
    return OutputConfig(
        dataset_path=str(tmp_path / "dataset.npz"),
        raw_signals_path=str(tmp_path / "signals_raw.npz"),
        results_dir=str(tmp_path / "results/"),
    )


@pytest.fixture
def builder(dataset_cfg, output_cfg):
    return DatasetBuilder(dataset_cfg, output_cfg)


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    X = rng.random((N, W))  # noqa: N806
    C = np.zeros((N, 5))  # noqa: N806
    idx = rng.integers(0, 5, N)
    C[np.arange(N), idx] = 1.0
    y = rng.random((N, 1))
    return X, C, y


@pytest.fixture
def split(builder, arrays):
    X, C, y = arrays  # noqa: N806
    return builder.split(X, C, y)


# ── Split sizes ───────────────────────────────────────────────────────────────

def test_dataset_builder_instantiates(dataset_cfg, output_cfg):
    b = DatasetBuilder(dataset_cfg, output_cfg)
    assert b is not None


def test_split_sizes_sum_to_total(split, arrays):
    X, _, _ = arrays  # noqa: N806
    total = split.X_train.shape[0] + split.X_val.shape[0] + split.X_test.shape[0]
    assert total == N


def test_train_size_approx_70_percent(split):
    assert split.X_train.shape[0] == pytest.approx(int(N * 0.70), abs=1)


def test_val_size_approx_15_percent(split):
    assert split.X_val.shape[0] == pytest.approx(int(N * 0.15), abs=1)


def test_test_size_approx_15_percent(split):
    n_train = int(N * 0.70)
    n_val = int(N * 0.15)
    assert split.X_test.shape[0] == N - n_train - n_val


def test_split_is_chronological(split, arrays):
    """First train samples must come before val samples in the original array."""
    X, _, _ = arrays  # noqa: N806
    n_train = split.X_train.shape[0]
    assert np.allclose(split.X_train[0], X[0])
    assert np.allclose(split.X_val[0], X[n_train])


# ── save_dataset ──────────────────────────────────────────────────────────────

def test_save_dataset_creates_npz_file(builder, split):
    path = builder.save_dataset(split)
    assert path.exists()


def test_dataset_npz_has_x_train_key(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert "X_train" in data


def test_dataset_npz_has_x_val_key(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert "X_val" in data


def test_dataset_npz_has_x_test_key(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert "X_test" in data


def test_dataset_npz_has_c_train_key(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert "C_train" in data


def test_dataset_npz_has_c_val_key(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert "C_val" in data


def test_dataset_npz_has_c_test_key(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert "C_test" in data


def test_dataset_npz_has_y_train_key(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert "y_train" in data


def test_dataset_npz_has_y_val_key(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert "y_val" in data


def test_dataset_npz_has_y_test_key(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert "y_test" in data


# ── Shape checks ──────────────────────────────────────────────────────────────

def test_x_train_shape_second_dim_equals_w(split):
    assert split.X_train.shape[1] == W


def test_x_val_shape_second_dim_equals_w(split):
    assert split.X_val.shape[1] == W


def test_x_test_shape_second_dim_equals_w(split):
    assert split.X_test.shape[1] == W


def test_c_train_shape_second_dim_equals_5(split):
    assert split.C_train.shape[1] == 5


def test_c_val_shape_second_dim_equals_5(split):
    assert split.C_val.shape[1] == 5


def test_c_test_shape_second_dim_equals_5(split):
    assert split.C_test.shape[1] == 5


def test_y_train_shape_second_dim_equals_1(split):
    assert split.y_train.shape[1] == 1


def test_y_val_shape_second_dim_equals_1(split):
    assert split.y_val.shape[1] == 1


def test_y_test_shape_second_dim_equals_1(split):
    assert split.y_test.shape[1] == 1


def test_y_train_shape_is_2d(split):
    assert split.y_train.ndim == 2 and split.y_train.shape[1] == 1


def test_c_train_rows_are_valid_one_hot(split):
    assert np.all(np.sum(split.C_train, axis=1) == 1)


def test_c_train_every_row_exactly_one_1(split):
    assert np.all(np.sum(split.C_train != 0, axis=1) == 1)


# ── NaN / Inf ─────────────────────────────────────────────────────────────────

def test_no_nan_in_saved_x_train(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert not np.any(np.isnan(data["X_train"]))


def test_no_nan_in_saved_y_train(builder, split):
    path = builder.save_dataset(split)
    data = np.load(path)
    assert not np.any(np.isnan(data["y_train"]))


# ── save_raw_signals ──────────────────────────────────────────────────────────

@pytest.fixture
def raw_data():
    n = 100
    t = np.linspace(0, 0.1, n, endpoint=False)
    clean = {k: np.ones(n) * i for i, k in enumerate(["s1", "s2", "s3", "s4", "s5"])}
    noisy = {k: v + 0.01 for k, v in clean.items()}
    return clean, noisy, t


def test_save_raw_signals_creates_npz_file(builder, raw_data):
    clean, noisy, t = raw_data
    path = builder.save_raw_signals(clean, noisy, t)
    assert path.exists()


def test_raw_npz_has_clean_signals_key(builder, raw_data):
    clean, noisy, t = raw_data
    path = builder.save_raw_signals(clean, noisy, t)
    data = np.load(path)
    assert "clean_signals" in data


def test_raw_npz_has_noisy_signals_key(builder, raw_data):
    clean, noisy, t = raw_data
    path = builder.save_raw_signals(clean, noisy, t)
    data = np.load(path)
    assert "noisy_signals" in data


def test_raw_npz_has_time_axis_key(builder, raw_data):
    clean, noisy, t = raw_data
    path = builder.save_raw_signals(clean, noisy, t)
    data = np.load(path)
    assert "time_axis" in data


def test_clean_signals_shape_in_raw_npz(builder, raw_data):
    clean, noisy, t = raw_data
    path = builder.save_raw_signals(clean, noisy, t)
    data = np.load(path)
    assert data["clean_signals"].shape == (5, len(t))


def test_noisy_signals_shape_in_raw_npz(builder, raw_data):
    clean, noisy, t = raw_data
    path = builder.save_raw_signals(clean, noisy, t)
    data = np.load(path)
    assert data["noisy_signals"].shape == (5, len(t))


def test_time_axis_shape_in_raw_npz(builder, raw_data):
    clean, noisy, t = raw_data
    path = builder.save_raw_signals(clean, noisy, t)
    data = np.load(path)
    assert data["time_axis"].shape == (len(t),)


# ── Ratio checks ──────────────────────────────────────────────────────────────

def test_split_train_ratio_from_config(split):
    assert split.X_train.shape[0] / N == pytest.approx(0.70, abs=1 / N)


def test_split_val_ratio_from_config(split):
    assert split.X_val.shape[0] / N == pytest.approx(0.15, abs=1 / N)


def test_split_test_ratio_from_config(split):
    assert split.X_test.shape[0] / N == pytest.approx(0.15, abs=2 / N)


def test_split_chronological_no_overlap(split, arrays):
    X, _, _ = arrays  # noqa: N806
    n_train = split.X_train.shape[0]
    n_val = split.X_val.shape[0]
    assert np.allclose(split.X_val[0], X[n_train])
    assert np.allclose(split.X_test[0], X[n_train + n_val])
