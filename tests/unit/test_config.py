"""Unit tests for shared/config.py — ConfigManager and validation."""

import json
import math

import pytest

from signal_dataset.shared.config import ConfigManager, ConfigValidationError

# ── Happy path: dataset section ──────────────────────────────────────────────

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


# ── Signals ───────────────────────────────────────────────────────────────────

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


# ── Noise: Gaussian ───────────────────────────────────────────────────────────

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


# ── Noise: Burst ──────────────────────────────────────────────────────────────

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


# ── Error cases ───────────────────────────────────────────────────────────────

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


def test_config_raises_on_split_ratios_not_summing_to_1(tmp_path, rate_limits_path, sample_config_dict):
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


def test_config_raises_on_burst_probability_above_1(tmp_path, rate_limits_path, sample_config_dict):
    sample_config_dict["noise"]["burst"]["probability"] = 1.5
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(sample_config_dict))
    with pytest.raises(ConfigValidationError):
        ConfigManager(p, rate_limits_path).load()


# ── Rate limits ───────────────────────────────────────────────────────────────

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
