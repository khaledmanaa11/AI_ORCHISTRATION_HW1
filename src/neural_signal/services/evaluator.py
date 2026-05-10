"""Evaluator — computes MSE on all splits and builds comparison table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_signal import constants as const


@dataclass
class EvalResult:
    """MSE scores for one model across all dataset splits."""

    model_name: str
    train_mse: float
    val_mse: float
    test_mse: float
    epochs_trained: int
    stopped_early: bool


class Evaluator:
    """Runs inference on all three splits and exports a comparison CSV.

    Loads the best checkpoint from disk before evaluating each model.
    """

    def __init__(self, results_dir: Path) -> None:
        """Initialise with the directory where comparison_table.csv will be saved."""
        self._results_dir = Path(results_dir)
        self._criterion = nn.MSELoss()

    def evaluate(
        self,
        model: nn.Module,
        model_name: str,
        checkpoint_path: Path,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs_trained: int,
        stopped_early: bool,
    ) -> EvalResult:
        """Load checkpoint, run inference on all splits, return EvalResult."""
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        return EvalResult(
            model_name=model_name,
            train_mse=self._compute_mse(model, train_loader),
            val_mse=self._compute_mse(model, val_loader),
            test_mse=self._compute_mse(model, test_loader),
            epochs_trained=epochs_trained,
            stopped_early=stopped_early,
        )

    def save_comparison_table(
        self, results: list[EvalResult], table_path: Path
    ) -> pd.DataFrame:
        """Build a DataFrame of all results and save to CSV."""
        rows = [
            {
                const.RESULT_COL_MODEL: r.model_name,
                const.RESULT_COL_TRAIN_MSE: r.train_mse,
                const.RESULT_COL_VAL_MSE: r.val_mse,
                const.RESULT_COL_TEST_MSE: r.test_mse,
                const.RESULT_COL_EPOCHS: r.epochs_trained,
                const.RESULT_COL_STOPPED_EARLY: r.stopped_early,
            }
            for r in results
        ]
        df = pd.DataFrame(rows)
        Path(table_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(table_path, index=False)
        return df

    @torch.no_grad()
    def _compute_mse(self, model: nn.Module, loader: DataLoader) -> float:
        """Compute mean MSE over all batches in loader."""
        total, n = 0.0, 0
        for xb, yb in loader:
            pred = model(xb)
            loss = self._criterion(pred, yb)
            total += loss.item() * len(xb)
            n += len(xb)
        return total / n if n > 0 else 0.0
