"""SensitivityAnalyzer — One-At-a-Time (OAT) parameter sensitivity analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SensitivityResult:
    """MSE scores for one OAT experiment."""

    param_name: str
    param_value: Any
    val_mse: float
    std_mse: float = 0.0


class SensitivityAnalyzer:
    """Run OAT sensitivity experiments over a list of hyperparameter values.

    For each value of a named parameter, trains a fresh model and records
    the best val MSE. All other hyperparameters are held at their baseline.
    """

    def __init__(self, base_config: dict, results_dir: Path) -> None:
        """Initialise with a baseline hyperparameter dict and output directory."""
        self._base = dict(base_config)
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        param_name: str,
        values: list,
        model_factory,
        loader_factory,
    ) -> list[SensitivityResult]:
        """Train one model per value and return list of SensitivityResult.

        Args:
            param_name: Key in base_config to vary.
            values: List of values to sweep.
            model_factory: Callable(config) → nn.Module.
            loader_factory: Callable(config) → (train_loader, val_loader).
        """
        import numpy as np
        folds = self._base.get("cross_validation_folds", 1)
        results = []
        for val in values:
            cfg = dict(self._base)
            cfg[param_name] = val
            fold_mses = []
            for fold in range(folds):
                torch.manual_seed(42 + fold)
                model = model_factory(cfg)
                train_loader, val_loader = loader_factory(cfg)
                val_mse = self._quick_train(model, train_loader, val_loader, cfg)
                fold_mses.append(val_mse)
            mean_mse = float(np.mean(fold_mses))
            std_mse = float(np.std(fold_mses)) if folds > 1 else 0.0
            results.append(SensitivityResult(param_name, val, mean_mse, std_mse))
        return results

    def _quick_train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
    ) -> float:
        """Train for a fixed number of epochs and return best val MSE."""
        lr = cfg.get("learning_rate", 1e-3)
        epochs = cfg.get("sensitivity_epochs", 5)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val = float("inf")

        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                total, n = 0.0, 0
                for xb, yb in val_loader:
                    loss = criterion(model(xb), yb)
                    total += loss.item() * len(xb)
                    n += len(xb)
            val_mse = total / n if n > 0 else 0.0
            if val_mse < best_val:
                best_val = val_mse

        return best_val

    def save_results(self, results: list[SensitivityResult]) -> Path:
        """Save OAT results to sensitivity_results.csv; return path."""
        path = self._results_dir / "sensitivity_results.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["param_name", "param_value", "val_mse", "std_mse"])
            writer.writeheader()
            for r in results:
                writer.writerow({"param_name": r.param_name,
                                 "param_value": r.param_value, "val_mse": r.val_mse, "std_mse": r.std_mse})
        return path

    def plot_line_chart(self, param_name: str, results: list[SensitivityResult]) -> Path:
        """Plot val_MSE vs param_value as a line chart; return PNG path."""
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r.param_value for r in results], [r.val_mse for r in results], "o-")
        ax.set(xlabel=param_name, ylabel="Val MSE", title=f"OAT: {param_name}")
        ax.grid(True, alpha=0.3)
        path = self._results_dir / f"sensitivity_{param_name}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_heatmap(self, results: list[SensitivityResult]) -> Path:
        """Plot param_value heatmap of val_MSE; return PNG path."""
        import matplotlib.pyplot as plt
        import numpy as np
        plt.switch_backend("Agg")
        mses = np.array([r.val_mse for r in results]).reshape(1, -1)
        params = [r.param_value for r in results]
        fig, ax = plt.subplots(figsize=(max(6, len(params)), 3))
        im = ax.imshow(mses, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels([str(p) for p in params], rotation=45, ha="right")
        ax.set_yticks([0])
        ax.set_yticklabels([results[0].param_name if results else ""])
        fig.colorbar(im, ax=ax, label="Val MSE")
        path = self._results_dir / "sensitivity_heatmap.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def make_simple_loaders(
        n_train: int = 200, n_val: int = 50, n_features: int = 15, batch_size: int = 32
    ) -> tuple[DataLoader, DataLoader]:
        """Create synthetic FCN-shape loaders for sensitivity testing."""
        x_tr = torch.randn(n_train, n_features)
        y_tr = torch.randn(n_train, 1)
        x_va = torch.randn(n_val, n_features)
        y_va = torch.randn(n_val, 1)
        train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=batch_size)
        return train_loader, val_loader
