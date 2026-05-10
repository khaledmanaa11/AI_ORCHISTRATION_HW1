"""Tests for neural_signal/shared/config.py — ConfigManager (neural_signal)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_signal.shared.config import (
    AppConfig,
    ConfigManager,
    ConfigValidationError,
)


def test_nn_config_loads_without_error(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert isinstance(cfg, AppConfig)


def test_nn_config_batch_size_64(config_path, rate_limits_path):
    assert ConfigManager(config_path, rate_limits_path).load().training.batch_size == 64


def test_nn_config_lr_0_001(config_path, rate_limits_path):
    assert ConfigManager(config_path, rate_limits_path).load().training.learning_rate == pytest.approx(0.001)


def test_nn_config_max_epochs_200(config_path, rate_limits_path):
    assert ConfigManager(config_path, rate_limits_path).load().training.max_epochs == 200


def test_nn_config_fcn_hidden_sizes(config_path, rate_limits_path):
    assert ConfigManager(config_path, rate_limits_path).load().fcn.hidden_sizes == [128, 64]


def test_nn_config_fcn_dropout(config_path, rate_limits_path):
    assert ConfigManager(config_path, rate_limits_path).load().fcn.dropout_rate == pytest.approx(0.1)


def test_nn_config_rnn_hidden_size(config_path, rate_limits_path):
    assert ConfigManager(config_path, rate_limits_path).load().rnn.hidden_size == 64


def test_nn_config_rnn_nonlinearity_tanh(config_path, rate_limits_path):
    assert ConfigManager(config_path, rate_limits_path).load().rnn.nonlinearity == "tanh"


def test_nn_config_lstm_hidden_size(config_path, rate_limits_path):
    assert ConfigManager(config_path, rate_limits_path).load().lstm.hidden_size == 64


def test_nn_config_lstm_dense_hidden_32(config_path, rate_limits_path):
    assert ConfigManager(config_path, rate_limits_path).load().lstm.dense_hidden_size == 32


def test_nn_config_raises_missing_file(tmp_path, rate_limits_path):
    with pytest.raises(FileNotFoundError):
        ConfigManager(tmp_path / "nope.json", rate_limits_path).load()


def test_nn_config_raises_negative_lr(config_path, rate_limits_path, tmp_path):
    raw = json.loads(Path(config_path).read_text())
    raw["training"]["learning_rate"] = -1
    p = tmp_path / "s.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_nn_config_raises_zero_batch(config_path, rate_limits_path, tmp_path):
    raw = json.loads(Path(config_path).read_text())
    raw["training"]["batch_size"] = 0
    p = tmp_path / "s.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_nn_config_raises_bad_val_split(config_path, rate_limits_path, tmp_path):
    raw = json.loads(Path(config_path).read_text())
    raw["training"]["val_split"] = 2.0
    p = tmp_path / "s.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_nn_config_raises_empty_hidden_sizes(config_path, rate_limits_path, tmp_path):
    raw = json.loads(Path(config_path).read_text())
    raw["fcn"]["hidden_sizes"] = []
    p = tmp_path / "s.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()
