"""Unit tests for services/noise_injector.py — NoiseInjector."""

import numpy as np
import pytest

from signal_dataset.services.noise_injector import NoiseInjector
from signal_dataset.shared.config import (
    BurstNoiseConfig,
    DatasetConfig,
    GaussianNoiseConfig,
    NoiseConfig,
    SignalConfig,
)


@pytest.fixture
def dataset_cfg():
    return DatasetConfig(
        duration_sec=1,
        sample_rate_hz=1000,
        window_size=10,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )


@pytest.fixture
def noise_cfg():
    return NoiseConfig(
        gaussian=GaussianNoiseConfig(
            sigma_amp=[0.05, 0.05, 0.03, 0.02],
            sigma_phase=[0.05, 0.05, 0.03, 0.02],
        ),
        burst=BurstNoiseConfig(
            probability=0.01,
            duration_range_samples=[5, 20],
            amp_magnitude=0.5,
            phase_magnitude=0.3,
        ),
    )


@pytest.fixture
def signal_cfgs():
    return [
        SignalConfig(id="s1", amplitude=2.0, frequency_hz=5, phase_rad=0),
        SignalConfig(id="s2", amplitude=1.5, frequency_hz=15, phase_rad=0.7854),
        SignalConfig(id="s3", amplitude=0.8, frequency_hz=50, phase_rad=0),
        SignalConfig(id="s4", amplitude=0.3, frequency_hz=100, phase_rad=0),
    ]


@pytest.fixture
def t(dataset_cfg):
    n = int(dataset_cfg.duration_sec * dataset_cfg.sample_rate_hz)
    return np.linspace(0, dataset_cfg.duration_sec, n, endpoint=False)


@pytest.fixture
def clean_signals(signal_cfgs, t):
    from signal_dataset.constants import TWO_PI
    signals = {}
    for s in signal_cfgs:
        signals[s.id] = s.amplitude * np.sin(TWO_PI * s.frequency_hz * t + s.phase_rad)
    signals["s5"] = sum(signals[k] for k in ["s1", "s2", "s3", "s4"])
    return signals


@pytest.fixture
def injector(noise_cfg, dataset_cfg):
    return NoiseInjector(noise_cfg, dataset_cfg, seed=42)


N = 1000  # matches fixture duration=1s at 1000 Hz


# ── Instantiation ─────────────────────────────────────────────────────────────

def test_noise_injector_instantiates(noise_cfg, dataset_cfg):
    inj = NoiseInjector(noise_cfg, dataset_cfg, seed=42)
    assert inj is not None


# ── Gaussian noise helpers ────────────────────────────────────────────────────

def test_generate_gaussian_amp_noise_shape(injector):
    n = injector._generate_gaussian_noise(0.05, N)
    assert n.shape == (N,)


def test_generate_gaussian_amp_noise_mean_near_zero(injector):
    n = injector._generate_gaussian_noise(0.05, N)
    assert abs(np.mean(n)) < 3 * 0.05 / np.sqrt(N)


def test_generate_gaussian_amp_noise_std_near_sigma(injector):
    sigma = 0.05
    n = injector._generate_gaussian_noise(sigma, N)
    assert np.std(n) == pytest.approx(sigma, rel=0.10)


def test_generate_gaussian_phase_noise_shape(injector):
    n = injector._generate_gaussian_noise(0.05, N)
    assert n.shape == (N,)


def test_generate_gaussian_phase_noise_mean_near_zero(injector):
    n = injector._generate_gaussian_noise(0.05, N)
    assert abs(np.mean(n)) < 3 * 0.05 / np.sqrt(N)


def test_generate_gaussian_phase_noise_std_near_sigma(injector):
    sigma = 0.03
    n = injector._generate_gaussian_noise(sigma, N)
    assert np.std(n) == pytest.approx(sigma, rel=0.10)


# ── Burst mask ────────────────────────────────────────────────────────────────

def test_generate_burst_mask_is_binary(injector):
    mask = injector._generate_burst_mask(N)
    assert set(np.unique(mask)).issubset({0, 1})


def test_generate_burst_mask_shape(injector):
    mask = injector._generate_burst_mask(N)
    assert mask.shape == (N,)


def test_generate_burst_mask_burst_durations_in_range(injector):
    mask = injector._generate_burst_mask(10000)
    # Find run lengths of 1s and verify they are within [5, 20]
    runs = []
    count = 0
    for val in mask:
        if val == 1:
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    for r in runs:
        assert 5 <= r <= 20


def test_generate_burst_noise_amp_shape(injector):
    mask = injector._generate_burst_mask(N)
    noise = injector._generate_burst_noise(mask, 0.5)
    assert noise.shape == (N,)


def test_generate_burst_noise_phase_shape(injector):
    mask = injector._generate_burst_mask(N)
    noise = injector._generate_burst_noise(mask, 0.3)
    assert noise.shape == (N,)


# ── inject_single ─────────────────────────────────────────────────────────────

def test_inject_single_differs_from_clean(injector, clean_signals, signal_cfgs, t):
    noisy = injector.inject_single(clean_signals["s1"], signal_cfgs[0], 0, t)
    assert not np.allclose(noisy, clean_signals["s1"])


def test_inject_single_shape_preserved(injector, clean_signals, signal_cfgs, t):
    noisy = injector.inject_single(clean_signals["s1"], signal_cfgs[0], 0, t)
    assert noisy.shape == clean_signals["s1"].shape


def test_inject_single_no_nan(injector, clean_signals, signal_cfgs, t):
    noisy = injector.inject_single(clean_signals["s1"], signal_cfgs[0], 0, t)
    assert not np.any(np.isnan(noisy))


def test_inject_single_no_inf(injector, clean_signals, signal_cfgs, t):
    noisy = injector.inject_single(clean_signals["s1"], signal_cfgs[0], 0, t)
    assert not np.any(np.isinf(noisy))


# ── inject_all ────────────────────────────────────────────────────────────────

def test_inject_all_returns_dict_with_5_keys(injector, clean_signals, signal_cfgs, t):
    result = injector.inject_all(clean_signals, signal_cfgs, t)
    assert len(result) == 5


def test_inject_all_keys_match_signal_ids(injector, clean_signals, signal_cfgs, t):
    result = injector.inject_all(clean_signals, signal_cfgs, t)
    assert set(result.keys()) == {"s1", "s2", "s3", "s4", "s5"}


def test_inject_all_values_are_ndarrays(injector, clean_signals, signal_cfgs, t):
    result = injector.inject_all(clean_signals, signal_cfgs, t)
    for v in result.values():
        assert isinstance(v, np.ndarray)


# ── Statistical properties ────────────────────────────────────────────────────

def test_n_amp_and_n_phase_are_independent(injector):
    n_amp = injector._generate_gaussian_noise(0.05, 10000)
    n_phase = injector._generate_gaussian_noise(0.05, 10000)
    corr = np.corrcoef(n_amp, n_phase)[0, 1]
    assert abs(corr) < 0.1


def test_reproducibility_with_same_seed(noise_cfg, dataset_cfg, clean_signals, signal_cfgs, t):
    i1 = NoiseInjector(noise_cfg, dataset_cfg, seed=7)
    i2 = NoiseInjector(noise_cfg, dataset_cfg, seed=7)
    r1 = i1.inject_all(clean_signals, signal_cfgs, t)
    r2 = i2.inject_all(clean_signals, signal_cfgs, t)
    assert np.allclose(r1["s1"], r2["s1"])


def test_different_seeds_give_different_noise(noise_cfg, dataset_cfg, clean_signals, signal_cfgs, t):
    i1 = NoiseInjector(noise_cfg, dataset_cfg, seed=1)
    i2 = NoiseInjector(noise_cfg, dataset_cfg, seed=2)
    r1 = i1.inject_all(clean_signals, signal_cfgs, t)
    r2 = i2.inject_all(clean_signals, signal_cfgs, t)
    assert not np.allclose(r1["s1"], r2["s1"])


def test_zero_sigma_gaussian_produces_no_gaussian_noise(noise_cfg, dataset_cfg, signal_cfgs, t):
    zero_noise_cfg = NoiseConfig(
        gaussian=GaussianNoiseConfig(sigma_amp=[0, 0, 0, 0], sigma_phase=[0, 0, 0, 0]),
        burst=BurstNoiseConfig(probability=0.0, duration_range_samples=[5, 20], amp_magnitude=0.5, phase_magnitude=0.3),
    )
    from signal_dataset.constants import TWO_PI
    clean = {}
    for s in signal_cfgs:
        clean[s.id] = s.amplitude * np.sin(TWO_PI * s.frequency_hz * t + s.phase_rad)
    clean["s5"] = sum(clean[k] for k in ["s1", "s2", "s3", "s4"])
    inj = NoiseInjector(zero_noise_cfg, dataset_cfg, seed=42)
    result = inj.inject_all(clean, signal_cfgs, t)
    assert np.allclose(result["s1"], clean["s1"], atol=1e-10)


def test_zero_burst_probability_produces_no_burst_events(noise_cfg, dataset_cfg):
    no_burst_cfg = NoiseConfig(
        gaussian=GaussianNoiseConfig(sigma_amp=[0, 0, 0, 0], sigma_phase=[0, 0, 0, 0]),
        burst=BurstNoiseConfig(probability=0.0, duration_range_samples=[5, 20], amp_magnitude=0.5, phase_magnitude=0.3),
    )
    inj = NoiseInjector(no_burst_cfg, dataset_cfg, seed=42)
    mask = inj._generate_burst_mask(1000)
    assert np.sum(mask) == 0


def test_burst_magnitude_bounded(injector):
    mask = np.ones(N, dtype=float)
    noise = injector._generate_burst_noise(mask, 0.5)
    assert np.all(np.abs(noise) <= 0.5 + 1e-10)


def test_composite_noisy_differs_from_clean_composite(injector, clean_signals, signal_cfgs, t):
    result = injector.inject_all(clean_signals, signal_cfgs, t)
    assert not np.allclose(result["s5"], clean_signals["s5"])


def test_inject_all_s5_noisy_equals_sum_of_s1_s2_s3_s4_noisy(
    noise_cfg, dataset_cfg, clean_signals, signal_cfgs, t
):
    inj = NoiseInjector(noise_cfg, dataset_cfg, seed=99)
    result = inj.inject_all(clean_signals, signal_cfgs, t)
    expected = result["s1"] + result["s2"] + result["s3"] + result["s4"]
    assert np.allclose(result["s5"], expected)
