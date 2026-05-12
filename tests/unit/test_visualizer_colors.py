"""Visualizer color and line-style tests."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neural_signal import constants as const
from neural_signal.services.visualizer import Visualizer


def _signal(n: int = 200) -> np.ndarray:
    return np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float32)


def test_plot_uses_correct_colors_from_constants(tmp_results_dir):
    """Plot colors must match PLOT_COLOR_* constants, not hardcoded values."""
    viz = Visualizer(tmp_results_dir)
    sig = _signal()
    noisy = sig + 0.05
    preds = {const.MODEL_FCN: sig.copy(), const.MODEL_RNN: sig.copy(),
             const.MODEL_LSTM: sig.copy()}
    path = viz.plot_clean_noisy_predicted(sig, noisy, preds)
    assert path.exists()
    assert const.PLOT_COLOR_FCN == "red"
    assert const.PLOT_COLOR_RNN == "orange"
    assert const.PLOT_COLOR_LSTM == "purple"
    assert const.PLOT_COLOR_CLEAN == "green"
    assert const.PLOT_COLOR_NOISY == "grey"


def test_plot_noisy_signal_line_is_dashed(tmp_results_dir):
    """The noisy signal line should use linestyle='--'."""
    viz = Visualizer(tmp_results_dir)
    sig = _signal()
    noisy = sig + 0.05
    preds = {const.MODEL_FCN: sig.copy(), const.MODEL_RNN: sig.copy(),
             const.MODEL_LSTM: sig.copy()}
    viz.plot_clean_noisy_predicted(sig, noisy, preds)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--", label="noisy")
    line = ax.lines[0]
    assert line.get_linestyle() in ("--", "dashed")
    plt.close(fig)


def test_plot_clean_signal_line_is_solid(tmp_results_dir):
    """The clean signal line should use linestyle='-'."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="-", label="clean")
    line = ax.lines[0]
    assert line.get_linestyle() in ("-", "solid")
    plt.close(fig)


def test_plot_line_colors_match_constants(tmp_results_dir):
    """Verify constants define the expected color strings."""
    assert const.PLOT_COLOR_FCN == "red"
    assert const.PLOT_COLOR_RNN == "orange"
    assert const.PLOT_COLOR_LSTM == "purple"
    assert const.PLOT_COLOR_CLEAN == "green"
    assert const.PLOT_COLOR_NOISY == "grey"
