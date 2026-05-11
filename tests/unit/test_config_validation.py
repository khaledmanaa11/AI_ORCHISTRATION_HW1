"""Unit tests for ConfigManager — error cases and rate limits."""

import json

import pytest

from signal_dataset.shared.config import ConfigManager, ConfigValidationError


def test_config_raises_on_missing_file(tmp_path, rate_limits_path):
    with pytest.raises(FileNotFoundError):
        ConfigManager(tmp_path / "missing.json", rate_limits_path).load()


def test_config_raises_on_invalid_json(tmp_path, rate_limits_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json}")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        ConfigManager(bad, rate_limits_path).load()


def test_config_raises_on_negative_duration(tmp_path, rate_limits_path, sample_config_dict):
    sample_config_dict["dataset"]["duration_sec"] = -1
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_config_raises_on_zero_sample_rate(tmp_path, rate_limits_path, sample_config_dict):
    sample_config_dict["dataset"]["sample_rate_hz"] = 0
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_config_raises_on_window_size_zero(tmp_path, rate_limits_path, sample_config_dict):
    sample_config_dict["dataset"]["window_size"] = 0
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_config_raises_on_split_ratios_not_summing_to_1(
    tmp_path, rate_limits_path, sample_config_dict
):
    sample_config_dict["dataset"]["train_ratio"] = 0.5
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_config_raises_on_negative_amplitude(tmp_path, rate_limits_path, sample_config_dict):
    sample_config_dict["signals"][0]["amplitude"] = -1.0
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_config_raises_on_negative_frequency(tmp_path, rate_limits_path, sample_config_dict):
    sample_config_dict["signals"][0]["frequency_hz"] = -5
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_config_raises_on_empty_signals_list(tmp_path, rate_limits_path, sample_config_dict):
    sample_config_dict["signals"] = []
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_config_raises_on_negative_noise_sigma(tmp_path, rate_limits_path, sample_config_dict):
    sample_config_dict["noise"]["gaussian"]["sigma_amp"] = [-0.1, 0.05, 0.03, 0.02]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_config_raises_on_burst_probability_above_1(
    tmp_path, rate_limits_path, sample_config_dict
):
    sample_config_dict["noise"]["burst"]["probability"] = 1.5
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


def test_config_loads_rate_limits_json(config_path, rate_limits_path):
    mgr = ConfigManager(config_path, rate_limits_path)
    mgr.load()
    assert mgr.rate_limits is not None


def test_config_rate_limit_default_service_exists(config_path, rate_limits_path):
    mgr = ConfigManager(config_path, rate_limits_path)
    mgr.load()
    assert "default" in mgr.rate_limits["rate_limits"]["services"]


def test_config_rate_limit_requests_per_minute_positive(config_path, rate_limits_path):
    mgr = ConfigManager(config_path, rate_limits_path)
    mgr.load()
    rpm = mgr.rate_limits["rate_limits"]["services"]["default"]["requests_per_minute"]
    assert rpm > 0
