"""Tests for services/preprocessor.py — Preprocessor."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neural_signal.services.preprocessor import Preprocessor


@pytest.fixture()
def x_train_fixture() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((200, 10)).astype(np.float32)


@pytest.fixture()
def x_val_fixture() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.standard_normal((50, 10)).astype(np.float32) + 5.0


@pytest.fixture()
def x_test_fixture() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.standard_normal((50, 10)).astype(np.float32) - 3.0


def test_preprocessor_instantiates(tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    assert p is not None


def test_fit_computes_mean_from_train_only(x_train_fixture, tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    params = p.fit(x_train_fixture)
    expected = x_train_fixture.mean(axis=0)
    np.testing.assert_allclose(params.mean, expected, atol=1e-5)


def test_fit_computes_std_from_train_only(x_train_fixture, tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    params = p.fit(x_train_fixture)
    expected = x_train_fixture.std(axis=0)
    np.testing.assert_allclose(params.std, np.where(expected == 0, 1.0, expected), atol=1e-5)


def test_transform_mean_of_normalized_train_approx_zero(x_train_fixture, tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    x_norm = p.fit_transform(x_train_fixture)
    np.testing.assert_allclose(x_norm.mean(axis=0), 0.0, atol=1e-5)


def test_transform_std_of_normalized_train_approx_one(x_train_fixture, tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    x_norm = p.fit_transform(x_train_fixture)
    np.testing.assert_allclose(x_norm.std(axis=0), 1.0, atol=1e-5)


def test_val_normalized_using_train_mean(x_train_fixture, x_val_fixture, tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    p.fit(x_train_fixture)
    x_val_norm = p.transform(x_val_fixture)
    # Val mean after normalisation with train stats should NOT be zero
    assert not np.allclose(x_val_norm.mean(axis=0), 0.0, atol=0.1)


def test_test_normalized_using_train_mean(x_train_fixture, x_test_fixture, tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    p.fit(x_train_fixture)
    x_test_norm = p.transform(x_test_fixture)
    assert not np.allclose(x_test_norm.mean(axis=0), 0.0, atol=0.1)


def test_fit_transform_returns_correct_shape(x_train_fixture, tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    out = p.fit_transform(x_train_fixture)
    assert out.shape == x_train_fixture.shape


def test_no_nan_after_normalization(x_train_fixture, tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    out = p.fit_transform(x_train_fixture)
    assert not np.isnan(out).any()


def test_no_inf_after_normalization(x_train_fixture, tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    out = p.fit_transform(x_train_fixture)
    assert not np.isinf(out).any()


def test_scaler_params_saved_to_json(x_train_fixture, tmp_path):
    path = tmp_path / "scaler.json"
    p = Preprocessor(path)
    p.fit_transform(x_train_fixture)
    p.save_params()
    assert path.exists()


def test_scaler_params_json_has_mean_key(x_train_fixture, tmp_path):
    path = tmp_path / "scaler.json"
    p = Preprocessor(path)
    p.fit_transform(x_train_fixture)
    p.save_params()
    raw = json.loads(path.read_text())
    assert "mean" in raw


def test_scaler_params_json_has_std_key(x_train_fixture, tmp_path):
    path = tmp_path / "scaler.json"
    p = Preprocessor(path)
    p.fit_transform(x_train_fixture)
    p.save_params()
    raw = json.loads(path.read_text())
    assert "std" in raw


def test_handles_zero_std_without_division_error(tmp_path):
    x_const = np.ones((50, 5), dtype=np.float32)  # all same → std=0
    p = Preprocessor(tmp_path / "scaler.json")
    out = p.fit_transform(x_const)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()


def test_raises_before_transform_without_fit(tmp_path):
    p = Preprocessor(tmp_path / "scaler.json")
    with pytest.raises(RuntimeError):
        p.transform(np.ones((10, 5), dtype=np.float32))
