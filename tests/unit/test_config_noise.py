"""Unit tests for ConfigManager — noise configuration section."""

import pytest

from signal_dataset.shared.config import ConfigManager


def test_config_noise_gaussian_sigma_amp_has_4_entries(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert len(cfg.noise.gaussian.sigma_amp) == 4


def test_config_noise_gaussian_sigma_amp_values_exact(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.noise.gaussian.sigma_amp == pytest.approx([0.05, 0.05, 0.03, 0.02])


def test_config_noise_gaussian_sigma_phase_has_4_entries(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert len(cfg.noise.gaussian.sigma_phase) == 4


def test_config_noise_gaussian_sigma_phase_values_exact(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.noise.gaussian.sigma_phase == pytest.approx([0.05, 0.05, 0.03, 0.02])


def test_config_noise_burst_probability_equals_0_01(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.noise.burst.probability == pytest.approx(0.01)


def test_config_noise_burst_duration_min_equals_5(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.noise.burst.duration_range_samples[0] == 5


def test_config_noise_burst_duration_max_equals_20(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.noise.burst.duration_range_samples[1] == 20


def test_config_noise_burst_duration_range_valid(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    lo, hi = cfg.noise.burst.duration_range_samples
    assert lo < hi and lo > 0


def test_config_noise_burst_amp_magnitude_equals_0_5(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.noise.burst.amp_magnitude == pytest.approx(0.5)


def test_config_noise_burst_phase_magnitude_equals_0_3(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.noise.burst.phase_magnitude == pytest.approx(0.3)
