"""Integration tests for the full dataset generation pipeline."""

import json

import numpy as np
import pytest

from signal_dataset.sdk.sdk import DatasetSDK


@pytest.fixture(scope="module")
def config_path():
    from pathlib import Path
    return Path(__file__).parent.parent.parent / "config" / "setup.json"


@pytest.fixture(scope="module")
def rate_limits_path():
    from pathlib import Path
    return Path(__file__).parent.parent.parent / "config" / "rate_limits.json"


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory, config_path, rate_limits_path):
    """Run the full pipeline once and cache the result for all integration tests."""
    out = tmp_path_factory.mktemp("data")
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["output"]["dataset_path"] = str(out / "dataset.npz")
    cfg["output"]["raw_signals_path"] = str(out / "signals_raw.npz")
    cfg["output"]["results_dir"] = str(out / "results/")
    mod_cfg = out / "setup_mod.json"
    mod_cfg.write_text(json.dumps(cfg))
    sdk = DatasetSDK(mod_cfg, rate_limits_path)
    return sdk.generate_dataset()


@pytest.fixture(scope="module")
def dataset_data(pipeline_result):
    return np.load(pipeline_result.dataset_path)


@pytest.fixture(scope="module")
def raw_data(pipeline_result):
    return np.load(pipeline_result.raw_path)


W = 10
N_TOTAL = 10000 - W + 1  # 9991


# ── Files exist ───────────────────────────────────────────────────────────────

def test_full_pipeline_runs_without_error(pipeline_result):
    assert pipeline_result is not None


def test_full_pipeline_creates_dataset_npz(pipeline_result):
    assert pipeline_result.dataset_path.exists()


def test_full_pipeline_creates_raw_signals_npz(pipeline_result):
    assert pipeline_result.raw_path.exists()


def test_dataset_npz_loadable(pipeline_result):
    data = np.load(pipeline_result.dataset_path)
    assert data is not None


def test_raw_npz_loadable(pipeline_result):
    data = np.load(pipeline_result.raw_path)
    assert data is not None


# ── Dtype checks ──────────────────────────────────────────────────────────────

def test_x_train_dtype_is_float(dataset_data):
    assert np.issubdtype(dataset_data["X_train"].dtype, np.floating)


def test_y_train_dtype_is_float(dataset_data):
    assert np.issubdtype(dataset_data["y_train"].dtype, np.floating)


def test_c_train_dtype_is_float_or_int(dataset_data):
    dtype = dataset_data["C_train"].dtype
    assert np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)


# ── NaN / Inf ─────────────────────────────────────────────────────────────────

def test_x_train_no_nan(dataset_data):
    assert not np.any(np.isnan(dataset_data["X_train"]))


def test_x_train_no_inf(dataset_data):
    assert not np.any(np.isinf(dataset_data["X_train"]))


def test_y_train_no_nan(dataset_data):
    assert not np.any(np.isnan(dataset_data["y_train"]))


def test_y_train_no_inf(dataset_data):
    assert not np.any(np.isinf(dataset_data["y_train"]))


def test_x_val_no_nan(dataset_data):
    assert not np.any(np.isnan(dataset_data["X_val"]))


def test_x_test_no_nan(dataset_data):
    assert not np.any(np.isnan(dataset_data["X_test"]))


def test_y_val_no_nan(dataset_data):
    assert not np.any(np.isnan(dataset_data["y_val"]))


def test_y_test_no_nan(dataset_data):
    assert not np.any(np.isnan(dataset_data["y_test"]))


# ── Shape checks ──────────────────────────────────────────────────────────────

def test_x_train_shape_second_dim_equals_w(dataset_data):
    assert dataset_data["X_train"].shape[1] == W


def test_c_train_shape_second_dim_equals_5(dataset_data):
    assert dataset_data["C_train"].shape[1] == 5


def test_y_train_shape_second_dim_equals_1(dataset_data):
    assert dataset_data["y_train"].shape[1] == 1


def test_c_train_rows_all_sum_to_1(dataset_data):
    assert np.all(np.sum(dataset_data["C_train"], axis=1) == 1)


# ── Source verification ───────────────────────────────────────────────────────

def test_x_train_windows_match_s5_noisy_slices(dataset_data, raw_data):
    s5_noisy = raw_data["noisy_signals"][4]  # index 4 = s5 in SIGNAL_IDS order
    x_train = dataset_data["X_train"]
    assert np.allclose(x_train[0], s5_noisy[0:W])


def test_y_train_label_matches_selected_clean_signal(dataset_data, raw_data):
    c_train = dataset_data["C_train"]
    y_train = dataset_data["y_train"]
    clean_signals = raw_data["clean_signals"]  # shape (5, N)
    for k in range(min(20, len(y_train))):
        i = int(np.argmax(c_train[k]))
        center = k + W // 2
        expected = clean_signals[i][center]
        assert y_train[k, 0] == pytest.approx(expected)


# ── Split sizes ───────────────────────────────────────────────────────────────

def test_train_val_test_sizes_sum_to_total_windows(dataset_data):
    n = dataset_data["X_train"].shape[0] + dataset_data["X_val"].shape[0] + dataset_data["X_test"].shape[0]
    assert n == N_TOTAL


def test_total_windows_equals_10000_minus_w_plus_1(dataset_data):
    n = dataset_data["X_train"].shape[0] + dataset_data["X_val"].shape[0] + dataset_data["X_test"].shape[0]
    assert n == N_TOTAL


def test_split_ratios_match_config_70_15_15(dataset_data):
    n_train = dataset_data["X_train"].shape[0]
    n_val = dataset_data["X_val"].shape[0]
    total = n_train + n_val + dataset_data["X_test"].shape[0]
    assert n_train / total == pytest.approx(0.70, abs=1 / total)
    assert n_val / total == pytest.approx(0.15, abs=1 / total)


# ── Raw signals ───────────────────────────────────────────────────────────────

def test_clean_signals_in_raw_npz_have_correct_frequencies(raw_data):
    clean = raw_data["clean_signals"]  # (5, 10000)
    fft = np.fft.rfft(clean[0])
    freqs = np.fft.rfftfreq(10000, d=0.001)
    dominant = freqs[np.argmax(np.abs(fft))]
    assert dominant == pytest.approx(5.0, abs=0.5)


def test_noisy_signals_differ_from_clean(raw_data):
    assert not np.allclose(raw_data["clean_signals"], raw_data["noisy_signals"])


def test_composite_noisy_in_raw_npz_equals_sum_of_noisy_components(raw_data):
    noisy = raw_data["noisy_signals"]
    s5_noisy = noisy[4]
    expected = noisy[0] + noisy[1] + noisy[2] + noisy[3]
    assert np.allclose(s5_noisy, expected)


def test_signals_raw_contains_5_clean_and_5_noisy(raw_data):
    assert raw_data["clean_signals"].shape == (5, 10000)
    assert raw_data["noisy_signals"].shape == (5, 10000)


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_dataset_reproducible_with_same_seed(tmp_path_factory, config_path, rate_limits_path):
    out1 = tmp_path_factory.mktemp("rep1")
    out2 = tmp_path_factory.mktemp("rep2")
    for out in (out1, out2):
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["output"]["dataset_path"] = str(out / "dataset.npz")
        cfg["output"]["raw_signals_path"] = str(out / "raw.npz")
        cfg["output"]["results_dir"] = str(out / "results/")
        mod = out / "setup.json"
        mod.write_text(json.dumps(cfg))
        DatasetSDK(mod, rate_limits_path).generate_dataset()
    d1 = np.load(out1 / "dataset.npz")
    d2 = np.load(out2 / "dataset.npz")
    assert np.array_equal(d1["X_train"], d2["X_train"])


# ── Model input ───────────────────────────────────────────────────────────────

def test_model_input_concat_shape_is_15(dataset_data):
    x_w = dataset_data["X_train"][0]
    c = dataset_data["C_train"][0]
    concat = np.concatenate([x_w, c])
    assert concat.shape == (15,)
