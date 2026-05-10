"""Unit tests for services/windower.py — Windower."""

import numpy as np
import pytest

from signal_dataset.constants import TWO_PI
from signal_dataset.services.windower import Windower

W = 10   # window size used across tests
N_SIG = 5  # number of signal classes


@pytest.fixture
def small_signals():
    """Minimal clean and noisy signal dicts for fast windowing tests."""
    n = 200
    t = np.linspace(0, 0.2, n, endpoint=False)
    clean = {
        "s1": 2.0 * np.sin(TWO_PI * 5 * t),
        "s2": 1.5 * np.sin(TWO_PI * 15 * t + 0.7854),
        "s3": 0.8 * np.sin(TWO_PI * 50 * t),
        "s4": 0.3 * np.sin(TWO_PI * 100 * t),
    }
    clean["s5"] = sum(clean[k] for k in ["s1", "s2", "s3", "s4"])
    rng = np.random.default_rng(0)
    noisy = {k: v + rng.normal(0, 0.01, n) for k, v in clean.items()}
    noisy["s5"] = sum(noisy[k] for k in ["s1", "s2", "s3", "s4"])
    return clean, noisy


@pytest.fixture
def windower():
    return Windower(window_size=W, n_signals=N_SIG, seed=42)


@pytest.fixture
def result(windower, small_signals):
    clean, noisy = small_signals
    return windower.build_windows(noisy, clean)


n_total = 200 - W + 1  # 191


# ── Instantiation ─────────────────────────────────────────────────────────────

def test_windower_instantiates():
    w = Windower(window_size=W, n_signals=N_SIG, seed=42)
    assert w is not None


# ── X shape & values ──────────────────────────────────────────────────────────

def test_build_windows_x_shape_is_n_windows_by_w(result):
    assert result.X.shape == (n_total, W)


def test_n_windows_equals_n_samples_minus_w_plus_1(result):
    assert result.X.shape[0] == n_total


def test_x_windows_sourced_from_s5_noisy_only(result, small_signals):
    _, noisy = small_signals
    assert np.allclose(result.X[0], noisy["s5"][0:W])
    assert np.allclose(result.X[5], noisy["s5"][5 : 5 + W])


def test_x_does_not_equal_s1_noisy_window(result, small_signals):
    _, noisy = small_signals
    assert not np.allclose(result.X[0], noisy["s1"][0:W])


def test_x_does_not_equal_s2_noisy_window(result, small_signals):
    _, noisy = small_signals
    assert not np.allclose(result.X[0], noisy["s2"][0:W])


def test_x_does_not_equal_s3_noisy_window(result, small_signals):
    _, noisy = small_signals
    assert not np.allclose(result.X[0], noisy["s3"][0:W])


def test_x_does_not_equal_s4_noisy_window(result, small_signals):
    _, noisy = small_signals
    assert not np.allclose(result.X[0], noisy["s4"][0:W])


# ── C shape & one-hot validity ────────────────────────────────────────────────

def test_c_shape_is_n_windows_by_5(result):
    assert result.C.shape == (n_total, 5)


def test_each_c_vector_is_one_hot(result):
    assert np.all(np.sum(result.C, axis=1) == 1)


def test_each_c_vector_has_exactly_one_nonzero(result):
    nonzero_per_row = np.sum(result.C != 0, axis=1)
    assert np.all(nonzero_per_row == 1)


def test_c_values_are_0_or_1(result):
    unique = np.unique(result.C)
    assert set(unique).issubset({0.0, 1.0})


def test_c_selects_signal_index_not_window_position(result):
    indices = np.argmax(result.C, axis=1)
    assert np.all(indices < 5)


def test_c_index_range_0_to_4(result):
    indices = np.argmax(result.C, axis=1)
    assert np.all((indices >= 0) & (indices <= 4))


# ── y shape & values ──────────────────────────────────────────────────────────

def test_y_shape_is_n_windows_by_1(result):
    assert result.y.shape == (n_total, 1)


def test_y_is_scalar_per_sample(result):
    assert result.y.ndim == 2 and result.y.shape[1] == 1


def test_y_matches_clean_value_of_c_selected_signal(result, small_signals):
    clean, _ = small_signals
    from signal_dataset.constants import SIGNAL_IDS
    for k in range(min(20, n_total)):
        i = int(np.argmax(result.C[k]))
        sid = SIGNAL_IDS[i]
        center = k + W // 2
        expected = clean[sid][center]
        assert result.y[k, 0] == pytest.approx(expected)


def test_y_shape_is_n_by_1_not_n(result):
    assert result.y.ndim == 2 and result.y.shape[1] == 1


def test_center_index_uses_integer_division(result, small_signals):
    clean, _ = small_signals
    from signal_dataset.constants import SIGNAL_IDS
    k = 0
    i = int(np.argmax(result.C[k]))
    sid = SIGNAL_IDS[i]
    center = k + W // 2   # floor division
    assert result.y[k, 0] == pytest.approx(clean[sid][center])


# ── NaN / Inf ─────────────────────────────────────────────────────────────────

def test_build_windows_no_nan_in_x(result):
    assert not np.any(np.isnan(result.X))


def test_build_windows_no_nan_in_y(result):
    assert not np.any(np.isnan(result.y))


def test_build_windows_no_inf_in_x(result):
    assert not np.any(np.isinf(result.X))


def test_build_windows_no_inf_in_y(result):
    assert not np.any(np.isinf(result.y))


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_reproducibility_same_seed_same_c(small_signals):
    clean, noisy = small_signals
    r1 = Windower(W, N_SIG, seed=7).build_windows(noisy, clean)
    r2 = Windower(W, N_SIG, seed=7).build_windows(noisy, clean)
    assert np.array_equal(r1.C, r2.C)


def test_different_seed_different_c(small_signals):
    clean, noisy = small_signals
    r1 = Windower(W, N_SIG, seed=1).build_windows(noisy, clean)
    r2 = Windower(W, N_SIG, seed=2).build_windows(noisy, clean)
    assert not np.array_equal(r1.C, r2.C)


# ── Error handling ────────────────────────────────────────────────────────────

def test_build_windows_raises_on_signal_shorter_than_window():
    short = {"s5": np.ones(5), "s1": np.ones(5), "s2": np.ones(5), "s3": np.ones(5), "s4": np.ones(5)}
    w = Windower(window_size=10, n_signals=5, seed=0)
    with pytest.raises(ValueError):
        w.build_windows(short, short)


# ── Coverage check ────────────────────────────────────────────────────────────

def test_all_signal_indices_appear_in_c_over_large_n():
    n = 2000
    t = np.linspace(0, 2, n, endpoint=False)
    clean = {
        "s1": np.sin(t), "s2": np.sin(t), "s3": np.sin(t), "s4": np.sin(t),
    }
    clean["s5"] = clean["s1"] + clean["s2"] + clean["s3"] + clean["s4"]
    noisy = {k: v + 0.01 for k, v in clean.items()}
    result = Windower(W, N_SIG, seed=0).build_windows(noisy, clean)
    indices = set(np.argmax(result.C, axis=1).tolist())
    assert indices == {0, 1, 2, 3, 4}


def test_model_input_concat_shape_is_w_plus_5(result):
    x_w = result.X[0]
    c = result.C[0]
    concat = np.concatenate([x_w, c])
    assert concat.shape == (W + 5,)


def test_windower_rejects_source_other_than_s5_noisy():
    w = Windower(W, N_SIG, seed=0)
    no_s5 = {"s1": np.ones(50), "s2": np.ones(50), "s3": np.ones(50), "s4": np.ones(50)}
    clean = dict(no_s5.items())
    clean["s5"] = np.ones(50)
    with pytest.raises(ValueError):
        w.build_windows(no_s5, clean)
