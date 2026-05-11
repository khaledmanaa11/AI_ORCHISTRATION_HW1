"""Unit tests for ConfigManager — dataset section and signals."""

import math

import pytest

from signal_dataset.shared.config import ConfigManager


def test_config_manager_loads_valid_json(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg is not None


def test_config_manager_returns_dataset_section(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.dataset is not None


def test_config_duration_sec_equals_10(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.dataset.duration_sec == 10


def test_config_sample_rate_hz_equals_1000(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.dataset.sample_rate_hz == 1000


def test_config_window_size_equals_10(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.dataset.window_size == 10


def test_config_train_ratio_equals_0_70(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.dataset.train_ratio == pytest.approx(0.70)


def test_config_val_ratio_equals_0_15(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.dataset.val_ratio == pytest.approx(0.15)


def test_config_test_ratio_equals_0_15(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.dataset.test_ratio == pytest.approx(0.15)


def test_config_split_ratios_sum_to_1(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    total = cfg.dataset.train_ratio + cfg.dataset.val_ratio + cfg.dataset.test_ratio
    assert total == pytest.approx(1.0, abs=1e-9)


def test_config_random_seed_equals_42(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.dataset.random_seed == 42


def test_config_signals_list_has_4_entries(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert len(cfg.signals) == 4


def test_config_s1_amplitude_equals_2_0(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[0].amplitude == pytest.approx(2.0)


def test_config_s1_frequency_equals_5(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[0].frequency_hz == 5


def test_config_s1_phase_equals_0(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[0].phase_rad == 0


def test_config_s2_amplitude_equals_1_5(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[1].amplitude == pytest.approx(1.5)


def test_config_s2_frequency_equals_15(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[1].frequency_hz == 15


def test_config_s2_phase_equals_0_7854(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[1].phase_rad == pytest.approx(0.7854)


def test_config_s2_phase_approx_pi_over_4(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[1].phase_rad == pytest.approx(math.pi / 4, abs=1e-4)


def test_config_s3_amplitude_equals_0_8(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[2].amplitude == pytest.approx(0.8)


def test_config_s3_frequency_equals_50(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[2].frequency_hz == 50


def test_config_s3_phase_equals_0(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[2].phase_rad == 0


def test_config_s4_amplitude_equals_0_3(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[3].amplitude == pytest.approx(0.3)


def test_config_s4_frequency_equals_100(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[3].frequency_hz == 100


def test_config_s4_phase_equals_0(config_path, rate_limits_path):
    cfg = ConfigManager(config_path, rate_limits_path).load()
    assert cfg.signals[3].phase_rad == 0
