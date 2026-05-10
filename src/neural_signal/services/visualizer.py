"""Visualizer — all result plots saved to results/ directory."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neural_signal import constants as const
from neural_signal.services.evaluator import EvalResult

matplotlib.use("Agg")  # non-interactive backend — safe for headless runs


class Visualizer:
    """Produces all required plots from training and evaluation data.

    All figures are saved to results_dir as PNG files and never shown
    interactively, so this class is safe to call from CLI scripts.
    """

    def __init__(self, results_dir: Path) -> None:
        """Initialise with the output directory for all PNG files."""
        self._dir = Path(results_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def plot_clean_noisy_predicted(
        self,
        clean: np.ndarray,
        noisy: np.ndarray,
        preds: dict[str, np.ndarray],
        n_samples: int = 300,
    ) -> Path:
        """Overlay clean, noisy, and all three model predictions on one figure."""
        fig, ax = plt.subplots(figsize=(14, 5))
        idx = np.arange(min(n_samples, len(clean)))
        ax.plot(idx, clean[idx], color=const.PLOT_COLOR_CLEAN, label="Clean", linewidth=1.5)
        ax.plot(idx, noisy[idx], color=const.PLOT_COLOR_NOISY, linestyle="--",
                label="Noisy", alpha=0.6)
        colors = {const.MODEL_FCN: const.PLOT_COLOR_FCN,
                  const.MODEL_RNN: const.PLOT_COLOR_RNN,
                  const.MODEL_LSTM: const.PLOT_COLOR_LSTM}
        for name, pred in preds.items():
            ax.plot(idx, pred[idx], color=colors.get(name, "blue"), label=name.upper())
        ax.set_title("Clean vs Noisy vs Predicted")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Signal value")
        ax.legend()
        out = self._dir / "clean_noisy_predicted.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_loss_curves(
        self,
        model_name: str,
        train_losses: list[float],
        val_losses: list[float],
    ) -> Path:
        """Plot train vs val MSE per epoch for one model."""
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Train MSE")
        ax.plot(epochs, val_losses, label="Val MSE")
        ax.set_title(f"Loss Curves — {model_name.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend()
        out = self._dir / f"loss_curves_{model_name}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_mse_comparison(self, results: list[EvalResult]) -> Path:
        """Grouped bar chart: train / val / test MSE for all three models."""
        names = [r.model_name.upper() for r in results]
        x = np.arange(len(names))
        w = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w, [r.train_mse for r in results], w, label="Train")
        ax.bar(x,     [r.val_mse   for r in results], w, label="Val")
        ax.bar(x + w, [r.test_mse  for r in results], w, label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("MSE")
        ax.set_title("MSE Comparison — All Models")
        ax.legend()
        out = self._dir / "mse_comparison.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_residuals(
        self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Path:
        """Residuals (y_pred - y_true) histogram for one model on the test set."""
        residuals = y_pred.flatten() - y_true.flatten()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(residuals, bins=40, edgecolor="black", alpha=0.75)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title(f"Residuals — {model_name.upper()}")
        ax.set_xlabel("Residual (pred − true)")
        ax.set_ylabel("Count")
        out = self._dir / f"residuals_{model_name}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_pred_vs_actual(
        self, predictions: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> Path:
        """Scatter plot of predicted vs actual for all three models (3 panels)."""
        n = len(predictions)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        colors = {const.MODEL_FCN: const.PLOT_COLOR_FCN,
                  const.MODEL_RNN: const.PLOT_COLOR_RNN,
                  const.MODEL_LSTM: const.PLOT_COLOR_LSTM}
        for ax, (name, (y_true, y_pred)) in zip(axes, predictions.items(), strict=False):
            ax.scatter(y_true.flatten(), y_pred.flatten(),
                       alpha=0.3, s=5, color=colors.get(name, "blue"))
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_title(name.upper())
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
        fig.suptitle("Predicted vs Actual")
        out = self._dir / "pred_vs_actual.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out
