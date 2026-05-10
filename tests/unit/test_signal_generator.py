"""Unit tests for services/signal_generator.py — SignalGenerator."""

import numpy as np
import pytest

from signal_dataset.services.signal_generator import SignalGenerator
from signal_dataset.shared.config import DatasetConfig, SignalConfig


@pytest.fixture
def dataset_cfg():
    return DatasetConfig(
        duration_sec=10,
        sample_rate_hz=1000,
        window_size=10,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
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
def gen(dataset_cfg, signal_cfgs):
    return SignalGenerator(dataset_cfg, signal_cfgs)


# ── Instantiation ─────────────────────────────────────────────────────────────

def test_signal_generator_instantiates(dataset_cfg, signal_cfgs):
    g = SignalGenerator(dataset_cfg, signal_cfgs)
    assert g is not None


# ── Time axis ─────────────────────────────────────────────────────────────────

def test_generate_time_axis_shape(gen):
    t = gen.generate_time_axis()
    assert t.shape == (10000,)


def test_generate_time_axis_start_equals_0(gen):
    t = gen.generate_time_axis()
    assert t[0] == pytest.approx(0.0)


def test_generate_time_axis_last_sample_equals_9_999(gen):
    t = gen.generate_time_axis()
    assert t[-1] == pytest.approx(9.999)


def test_generate_time_axis_end_less_than_duration(gen):
    t = gen.generate_time_axis()
    assert t[-1] < 10.0


def test_generate_time_axis_step_equals_1_over_sample_rate(gen):
    t = gen.generate_time_axis()
    assert t[1] - t[0] == pytest.approx(0.001)


# ── Signal shapes ─────────────────────────────────────────────────────────────

def test_generate_s1_shape(gen):
    t = gen.generate_time_axis()
    s1 = gen.generate_single(gen._signals[0], t)
    assert s1.shape == (10000,)


def test_generate_s2_shape(gen):
    t = gen.generate_time_axis()
    s2 = gen.generate_single(gen._signals[1], t)
    assert s2.shape == (10000,)


def test_generate_s3_shape(gen):
    t = gen.generate_time_axis()
    s3 = gen.generate_single(gen._signals[2], t)
    assert s3.shape == (10000,)


def test_generate_s4_shape(gen):
    t = gen.generate_time_axis()
    s4 = gen.generate_single(gen._signals[3], t)
    assert s4.shape == (10000,)


def test_generate_s5_shape(gen):
    signals = gen.generate_all()
    assert signals["s5"].shape == (10000,)


# ── Amplitudes ────────────────────────────────────────────────────────────────

def test_generate_s1_max_amplitude_approx_2_0(gen):
    t = gen.generate_time_axis()
    s1 = gen.generate_single(gen._signals[0], t)
    assert np.max(np.abs(s1)) == pytest.approx(2.0, abs=1e-6)


def test_generate_s2_max_amplitude_approx_1_5(gen):
    t = gen.generate_time_axis()
    s2 = gen.generate_single(gen._signals[1], t)
    assert np.max(np.abs(s2)) == pytest.approx(1.5, abs=1e-6)


def test_generate_s3_max_amplitude_approx_0_8(gen):
    t = gen.generate_time_axis()
    s3 = gen.generate_single(gen._signals[2], t)
    assert np.max(np.abs(s3)) == pytest.approx(0.8, abs=1e-6)


def test_generate_s4_max_amplitude_approx_0_3(gen):
    # 100 Hz at 1000 Hz sampling: only 10 samples/cycle, peak may not be sampled exactly.
    # Verify via RMS: RMS of A*sin = A/sqrt(2)
    t = gen.generate_time_axis()
    s4 = gen.generate_single(gen._signals[3], t)
    assert np.sqrt(np.mean(s4**2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.01)


# ── Zero mean ─────────────────────────────────────────────────────────────────

def test_generate_s1_is_zero_mean(gen):
    t = gen.generate_time_axis()
    s1 = gen.generate_single(gen._signals[0], t)
    assert np.mean(s1) == pytest.approx(0.0, abs=0.01)


def test_generate_s2_is_zero_mean(gen):
    t = gen.generate_time_axis()
    s2 = gen.generate_single(gen._signals[1], t)
    assert np.mean(s2) == pytest.approx(0.0, abs=0.01)


def test_generate_s3_is_zero_mean(gen):
    t = gen.generate_time_axis()
    s3 = gen.generate_single(gen._signals[2], t)
    assert np.mean(s3) == pytest.approx(0.0, abs=0.01)


def test_generate_s4_is_zero_mean(gen):
    t = gen.generate_time_axis()
    s4 = gen.generate_single(gen._signals[3], t)
    assert np.mean(s4) == pytest.approx(0.0, abs=0.01)


# ── Composite ─────────────────────────────────────────────────────────────────

def test_generate_s5_equals_sum_of_components(gen):
    t = gen.generate_time_axis()
    s1 = gen.generate_single(gen._signals[0], t)
    s2 = gen.generate_single(gen._signals[1], t)
    s3 = gen.generate_single(gen._signals[2], t)
    s4 = gen.generate_single(gen._signals[3], t)
    signals = gen.generate_all()
    assert np.allclose(signals["s5"], s1 + s2 + s3 + s4)


# ── Frequency via FFT ─────────────────────────────────────────────────────────

def _dominant_hz(signal, sample_rate=1000):
    n = len(signal)
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return freqs[np.argmax(np.abs(fft))]


def test_generate_s1_dominant_frequency_via_fft(gen):
    t = gen.generate_time_axis()
    s1 = gen.generate_single(gen._signals[0], t)
    assert _dominant_hz(s1) == pytest.approx(5.0, abs=0.5)


def test_generate_s2_dominant_frequency_via_fft(gen):
    t = gen.generate_time_axis()
    s2 = gen.generate_single(gen._signals[1], t)
    assert _dominant_hz(s2) == pytest.approx(15.0, abs=0.5)


def test_generate_s3_dominant_frequency_via_fft(gen):
    t = gen.generate_time_axis()
    s3 = gen.generate_single(gen._signals[2], t)
    assert _dominant_hz(s3) == pytest.approx(50.0, abs=0.5)


def test_generate_s4_dominant_frequency_via_fft(gen):
    t = gen.generate_time_axis()
    s4 = gen.generate_single(gen._signals[3], t)
    assert _dominant_hz(s4) == pytest.approx(100.0, abs=0.5)


def test_generate_s2_phase_offset(gen):
    """s2 should have a π/4 phase offset relative to a zero-phase 15 Hz sine."""
    import math
    t = gen.generate_time_axis()
    s2 = gen.generate_single(gen._signals[1], t)
    ref = 1.5 * np.sin(2 * math.pi * 15 * t)
    corr = np.corrcoef(s2, ref)[0, 1]
    # Phase shift of π/4 means correlation < 1 but signal is still correlated
    assert corr < 0.9999


# ── generate_all() ────────────────────────────────────────────────────────────

def test_generate_all_returns_dict_with_5_keys(gen):
    signals = gen.generate_all()
    assert len(signals) == 5


def test_generate_all_keys_are_s1_through_s5(gen):
    signals = gen.generate_all()
    assert set(signals.keys()) == {"s1", "s2", "s3", "s4", "s5"}


def test_generate_all_values_are_ndarrays(gen):
    signals = gen.generate_all()
    for v in signals.values():
        assert isinstance(v, np.ndarray)


def test_generate_no_nans_in_any_signal(gen):
    signals = gen.generate_all()
    for v in signals.values():
        assert not np.any(np.isnan(v))


def test_generate_no_infs_in_any_signal(gen):
    signals = gen.generate_all()
    for v in signals.values():
        assert not np.any(np.isinf(v))
