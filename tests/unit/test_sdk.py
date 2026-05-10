"""Unit tests for sdk/sdk.py — DatasetSDK."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from signal_dataset.sdk.sdk import DatasetResult, DatasetSDK
from signal_dataset.services.dataset_builder import SplitResult


@pytest.fixture
def sdk(config_path, rate_limits_path):
    """Real DatasetSDK using actual config files."""
    return DatasetSDK(config_path, rate_limits_path)


def _make_split() -> SplitResult:
    return SplitResult(
        X_train=np.zeros((70, 10)), C_train=np.zeros((70, 5)), y_train=np.zeros((70, 1)),
        X_val=np.zeros((15, 10)), C_val=np.zeros((15, 5)), y_val=np.zeros((15, 1)),
        X_test=np.zeros((15, 10)), C_test=np.zeros((15, 5)), y_test=np.zeros((15, 1)),
    )


def _make_mock_window_result() -> MagicMock:
    wr = MagicMock()
    wr.X = np.zeros((100, 10))
    wr.C = np.zeros((100, 5))
    wr.C[:, 0] = 1
    wr.y = np.zeros((100, 1))
    return wr


# ── Instantiation ─────────────────────────────────────────────────────────────

def test_sdk_instantiates_with_config_path(config_path, rate_limits_path):
    sdk = DatasetSDK(config_path, rate_limits_path)
    assert sdk is not None


def test_sdk_raises_config_error_on_bad_path(tmp_path, rate_limits_path):
    from signal_dataset.shared.config import ConfigValidationError
    with pytest.raises((FileNotFoundError, ConfigValidationError)):
        DatasetSDK(tmp_path / "missing.json", rate_limits_path)


def test_sdk_raises_config_error_on_invalid_json(tmp_path, rate_limits_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json}")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        DatasetSDK(bad, rate_limits_path)


# ── Version ───────────────────────────────────────────────────────────────────

def test_sdk_get_version_returns_string(sdk):
    assert isinstance(sdk.get_version(), str)


def test_sdk_get_version_starts_with_1(sdk):
    assert sdk.get_version().startswith("1")


# ── generate_dataset ──────────────────────────────────────────────────────────

def test_sdk_generate_dataset_returns_dataset_result(sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    result = sdk.generate_dataset()
    assert isinstance(result, DatasetResult)


def test_dataset_result_has_dataset_path_attribute(sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    result = sdk.generate_dataset()
    assert hasattr(result, "dataset_path")


def test_dataset_result_has_raw_path_attribute(sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    result = sdk.generate_dataset()
    assert hasattr(result, "raw_path")


def test_dataset_result_has_split_result_attribute(sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    result = sdk.generate_dataset()
    assert hasattr(result, "split_result")


def test_dataset_result_dataset_path_is_path_object(sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    result = sdk.generate_dataset()
    assert isinstance(result.dataset_path, Path)


def test_dataset_result_raw_path_is_path_object(sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    result = sdk.generate_dataset()
    assert isinstance(result.raw_path, Path)


def test_generate_dataset_creates_dataset_file(sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = sdk.generate_dataset()
    assert result.dataset_path.exists()


def test_generate_dataset_creates_raw_signals_file(sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = sdk.generate_dataset()
    assert result.raw_path.exists()


# ── Mock-based orchestration tests ────────────────────────────────────────────

def _run_sdk_with_mocks(config_path, rate_limits_path, tmp_path):
    """Helper: run generate_dataset() with all services fully mocked."""
    t = np.linspace(0, 10, 10000, endpoint=False)
    clean = {k: np.zeros(10000) for k in ["s1", "s2", "s3", "s4", "s5"]}
    noisy = {k: np.zeros(10000) for k in ["s1", "s2", "s3", "s4", "s5"]}
    sdk = DatasetSDK(config_path, rate_limits_path)
    with (
        patch("signal_dataset.sdk.sdk.SignalGenerator") as mock_gen,
        patch("signal_dataset.sdk.sdk.NoiseInjector") as mock_noise,
        patch("signal_dataset.sdk.sdk.Windower") as mock_wind,
        patch("signal_dataset.sdk.sdk.DatasetBuilder") as mock_builder,
    ):
        mock_gen.return_value.generate_time_axis.return_value = t
        mock_gen.return_value.generate_all.return_value = clean
        mock_noise.return_value.inject_all.return_value = noisy
        mock_wind.return_value.build_windows.return_value = _make_mock_window_result()
        mock_builder.return_value.split.return_value = _make_split()
        mock_builder.return_value.save_dataset.return_value = tmp_path / "dataset.npz"
        mock_builder.return_value.save_raw_signals.return_value = tmp_path / "raw.npz"
        sdk.generate_dataset()
        return mock_gen, mock_noise, mock_wind, mock_builder


def test_sdk_generate_calls_signal_generator(config_path, rate_limits_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_gen, _, _, _ = _run_sdk_with_mocks(config_path, rate_limits_path, tmp_path)
    mock_gen.assert_called_once()


def test_sdk_generate_calls_noise_injector(config_path, rate_limits_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, mock_noise, _, _ = _run_sdk_with_mocks(config_path, rate_limits_path, tmp_path)
    mock_noise.assert_called_once()


def test_sdk_generate_calls_windower(config_path, rate_limits_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, _, mock_wind, _ = _run_sdk_with_mocks(config_path, rate_limits_path, tmp_path)
    mock_wind.assert_called_once()


def test_sdk_generate_calls_dataset_builder(config_path, rate_limits_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, _, _, mock_builder = _run_sdk_with_mocks(config_path, rate_limits_path, tmp_path)
    mock_builder.assert_called_once()
