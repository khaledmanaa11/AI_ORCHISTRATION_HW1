"""SensitivityAnalyzer — One-At-a-Time (OAT) parameter sensitivity analysis."""

from __future__ import annotations

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
        results = []
        for val in values:
            cfg = dict(self._base)
            cfg[param_name] = val
            model = model_factory(cfg)
            train_loader, val_loader = loader_factory(cfg)
            val_mse = self._quick_train(model, train_loader, val_loader, cfg)
            results.append(SensitivityResult(param_name, val, val_mse))
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
