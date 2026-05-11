"""Tests for services/visualizer.py — Visualizer plot generation."""

from __future__ import annotations

import numpy as np
import pytest

from neural_signal.services.evaluator import EvalResult
from neural_signal.services.visualizer import Visualizer


@pytest.fixture()
def viz(tmp_results_dir) -> Visualizer:
    return Visualizer(tmp_results_dir)


@pytest.fixture()
def clean_signal() -> np.ndarray:
    return np.sin(np.linspace(0, 4 * np.pi, 400)).astype(np.float32)


@pytest.fixture()
def noisy_signal(clean_signal) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (clean_signal + rng.normal(0, 0.1, len(clean_signal))).astype(np.float32)


@pytest.fixture()
def preds(clean_signal) -> dict:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.05, len(clean_signal)).astype(np.float32)
    return {
        "fcn": clean_signal + noise,
        "rnn": clean_signal + noise * 0.8,
        "lstm": clean_signal + noise * 0.5,
    }


@pytest.fixture()
def eval_results() -> list[EvalResult]:
    return [
        EvalResult("fcn", 0.05, 0.06, 0.07, 10, False),
        EvalResult("rnn", 0.04, 0.05, 0.06, 12, False),
        EvalResult("lstm", 0.03, 0.04, 0.05, 15, True),
    ]


def test_visualizer_instantiates(tmp_results_dir):
    v = Visualizer(tmp_results_dir)
    assert v is not None


def test_plot_clean_noisy_predicted_creates_file(viz, clean_signal, noisy_signal, preds):
    path = viz.plot_clean_noisy_predicted(clean_signal, noisy_signal, preds)
    assert path.exists()


def test_plot_clean_noisy_predicted_is_png(viz, clean_signal, noisy_signal, preds):
    path = viz.plot_clean_noisy_predicted(clean_signal, noisy_signal, preds)
    assert path.suffix == ".png"


def test_plot_loss_curves_creates_file(viz):
    path = viz.plot_loss_curves("fcn", [0.5, 0.4, 0.3], [0.6, 0.5, 0.4])
    assert path.exists()


def test_plot_loss_curves_file_named_correctly(viz):
    path = viz.plot_loss_curves("rnn", [0.5], [0.6])
    assert "rnn" in path.name


def test_plot_mse_comparison_creates_file(viz, eval_results):
    path = viz.plot_mse_comparison(eval_results)
    assert path.exists()


def test_plot_residuals_creates_file(viz):
    y_true = np.random.randn(100).astype(np.float32)
    y_pred = y_true + np.random.randn(100).astype(np.float32) * 0.1
    path = viz.plot_residuals("fcn", y_true, y_pred)
    assert path.exists()


def test_plot_residuals_file_named_correctly(viz):
    y = np.zeros(50, dtype=np.float32)
    path = viz.plot_residuals("lstm", y, y)
    assert "lstm" in path.name


def test_plot_pred_vs_actual_creates_file(viz):
    y = np.random.randn(100).astype(np.float32)
    pred_dict = {
        "fcn": (y, y + 0.01),
        "rnn": (y, y + 0.02),
        "lstm": (y, y + 0.005),
    }
    path = viz.plot_pred_vs_actual(pred_dict)
    assert path.exists()


def test_results_dir_created_if_missing(tmp_path):
    Visualizer(tmp_path / "new_dir")
    assert (tmp_path / "new_dir").exists()


def test_plot_all_models_in_clean_noisy_file(viz, clean_signal, noisy_signal, preds):
    path = viz.plot_clean_noisy_predicted(clean_signal, noisy_signal, preds)
    assert path.stat().st_size > 5000  # non-trivial PNG


def test_clean_noisy_predicted_uses_correct_colors(
    tmp_results_dir, clean_signal, noisy_signal, preds
):
    """Clean line uses green, noisy line uses grey as defined in constants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neural_signal import constants as const

    fig, ax = plt.subplots()
    ax.plot(clean_signal[:10], color=const.PLOT_COLOR_CLEAN, label="Clean")
    ax.plot(noisy_signal[:10], color=const.PLOT_COLOR_NOISY, linestyle="--", label="Noisy")
    lines = ax.get_lines()
    assert lines[0].get_color() == const.PLOT_COLOR_CLEAN
    assert lines[1].get_color() == const.PLOT_COLOR_NOISY
    plt.close(fig)


def test_noisy_line_uses_dashed_style(tmp_results_dir, clean_signal, noisy_signal):
    """Noisy signal is drawn with a dashed line style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neural_signal import constants as const

    fig, ax = plt.subplots()
    ax.plot(noisy_signal[:10], color=const.PLOT_COLOR_NOISY, linestyle="--")
    line = ax.get_lines()[0]
    assert line.get_linestyle() == "--"
    plt.close(fig)


def test_model_colors_are_distinct(tmp_results_dir):
    """FCN, RNN, LSTM prediction colors must all be different."""
    from neural_signal import constants as const
    colors = [const.PLOT_COLOR_FCN, const.PLOT_COLOR_RNN, const.PLOT_COLOR_LSTM]
    assert len(set(colors)) == 3
