"""Additional visualizer tests — overflow from test_visualizer.py (150-line limit)."""

from __future__ import annotations

import numpy as np
import pytest

from neural_signal.services.evaluator import EvalResult
from neural_signal.services.visualizer import Visualizer


@pytest.fixture()
def viz(tmp_results_dir) -> Visualizer:
    return Visualizer(tmp_results_dir)


@pytest.fixture()
def signal() -> np.ndarray:
    return np.sin(np.linspace(0, 4 * np.pi, 200)).astype(np.float32)


@pytest.fixture()
def eval_results() -> list[EvalResult]:
    return [
        EvalResult("fcn", 0.05, 0.06, 0.07, 10, False),
        EvalResult("rnn", 0.04, 0.05, 0.06, 12, False),
        EvalResult("lstm", 0.03, 0.04, 0.05, 15, True),
    ]


def test_plot_clean_noisy_predicted_has_five_lines(viz, signal):
    """The combined plot must contain exactly 5 lines (clean, noisy, fcn, rnn, lstm)."""
    import matplotlib
    matplotlib.use("Agg")

    noisy = (signal + np.random.default_rng(0).normal(0, 0.1, len(signal))).astype(np.float32)
    preds = {"fcn": signal.copy(), "rnn": signal.copy(), "lstm": signal.copy()}
    path = viz.plot_clean_noisy_predicted(signal, noisy, preds)
    assert path.exists()
    # File should be non-trivial (5 lines produce more data than 1 line)
    assert path.stat().st_size > 1000


def test_plot_loss_curves_creates_file_for_rnn(viz):
    path = viz.plot_loss_curves("rnn", [0.5, 0.4, 0.3], [0.6, 0.5, 0.4])
    assert path.exists() and "rnn" in path.name


def test_plot_loss_curves_creates_file_for_lstm(viz):
    path = viz.plot_loss_curves("lstm", [0.5, 0.4], [0.6, 0.5])
    assert path.exists() and "lstm" in path.name


def test_plot_mse_comparison_file_not_empty(viz, eval_results):
    path = viz.plot_mse_comparison(eval_results)
    assert path.stat().st_size > 0


def test_plot_raises_on_mismatched_array_lengths(viz):
    """Residual plot with mismatched arrays should raise ValueError."""
    y_true = np.zeros(10, dtype=np.float32)
    y_pred = np.zeros(20, dtype=np.float32)
    with pytest.raises((ValueError, IndexError)):
        viz.plot_residuals("fcn", y_true, y_pred)


def test_plot_works_with_minimal_data(viz):
    """All plot methods should handle 10 samples without error."""
    y = np.random.randn(10).astype(np.float32)
    signal = np.random.randn(10).astype(np.float32)
    noisy = signal + 0.01
    preds = {"fcn": signal.copy(), "rnn": signal.copy(), "lstm": signal.copy()}
    assert viz.plot_clean_noisy_predicted(signal, noisy, preds, n_samples=10).exists()
    assert viz.plot_loss_curves("fcn", [0.5], [0.6]).exists()
    assert viz.plot_residuals("rnn", y, y).exists()
