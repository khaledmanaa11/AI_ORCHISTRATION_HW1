"""Visualisation orchestration helpers — extracted mixin for NeuralSignalSDK."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_signal import constants as const
from neural_signal.services.data_loader import DataBundle
from neural_signal.services.evaluator import EvalResult
from neural_signal.services.visualizer import Visualizer
from neural_signal.shared.config import AppConfig


class VizMixin:
    """Mixin that adds visualisation orchestration to NeuralSignalSDK.

    Depends on _build_model_resources and _train_results being defined
    on the host class (NeuralSignalSDK).
    """

    def _run_visualizations(
        self,
        bundle: DataBundle,
        results: list[EvalResult],
        build_fn: Callable,
        cfg: AppConfig,
    ) -> None:
        """Generate all mandatory plots and save to results_dir."""
        viz = Visualizer(Path(cfg.output.results_dir))
        clean = bundle.y_train.flatten()
        noisy = bundle.X_train[:, 0].flatten()

        preds: dict[str, np.ndarray] = {}
        for name in const.MODEL_NAMES:
            model, _, _, ckpt, _ = build_fn(name, bundle, cfg)
            ckpt_path = Path(ckpt)
            if ckpt_path.exists():
                model.load_state_dict(
                    torch.load(ckpt_path, map_location="cpu", weights_only=True)
                )
            model.eval()
            loader = bundle.train_loader if name == const.MODEL_FCN else bundle.seq_train_loader
            preds[name] = self._predict_all(model, loader)

        viz.plot_clean_noisy_predicted(clean, noisy, preds)
        for name in const.MODEL_NAMES:
            tr = getattr(self, "_train_results", {}).get(name)
            if tr:
                viz.plot_loss_curves(name, tr.train_losses, tr.val_losses)
        viz.plot_mse_comparison(results)

        pred_vs_actual: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in const.MODEL_NAMES:
            model, _, _, ckpt, _ = build_fn(name, bundle, cfg)
            ckpt_path = Path(ckpt)
            if ckpt_path.exists():
                model.load_state_dict(
                    torch.load(ckpt_path, map_location="cpu", weights_only=True)
                )
            model.eval()
            loader = bundle.test_loader if name == const.MODEL_FCN else bundle.seq_test_loader
            y_pred = self._predict_all(model, loader)
            y_true = bundle.y_test.flatten()
            viz.plot_residuals(name, y_true, y_pred)
            pred_vs_actual[name] = (y_true, y_pred)

        viz.plot_pred_vs_actual(pred_vs_actual)

    @staticmethod
    @torch.no_grad()
    def _predict_all(model: torch.nn.Module, loader: DataLoader) -> np.ndarray:  # type: ignore[type-arg]
        """Run model on all batches in loader and return flat predictions array."""
        preds = []
        for xb, _ in loader:
            preds.append(model(xb).cpu().numpy())
        return np.concatenate(preds).flatten()
