"""Trainer — training loop with early stopping and checkpoint saving."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_signal.shared.config import TrainingConfig


@dataclass
class TrainingResult:
    """Summary of a completed training run."""

    best_val_mse: float
    epochs_trained: int
    stopped_early: bool
    checkpoint_path: str
    train_losses: list[float]
    val_losses: list[float]


class Trainer:
    """Epoch loop with Adam optimiser, MSE loss, early stopping, and checkpoint saving.

    Saves the best model weights (lowest val MSE) to checkpoint_path.
    Logs per-epoch train/val MSE to a CSV alongside the checkpoint.
    """

    def __init__(self, cfg: TrainingConfig, checkpoint_path: Path, log_path: Path) -> None:
        """Initialise with training config and output paths."""
        self._cfg = cfg
        self._ckpt = Path(checkpoint_path)
        self._log = Path(log_path)
        self._criterion = nn.MSELoss()

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        weight_decay: float = 0.0,
    ) -> TrainingResult:
        """Run the full training loop and return a TrainingResult."""
        device = torch.device(self._cfg.device)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self._cfg.learning_rate, weight_decay=weight_decay
        )
        best_val = float("inf")
        patience_counter = 0
        train_losses: list[float] = []
        val_losses: list[float] = []
        epochs_done = 0

        self._ckpt.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self._cfg.max_epochs + 1):
            train_mse = self._run_epoch(model, train_loader, optimizer, device, training=True)
            val_mse = self._run_epoch(model, val_loader, None, device, training=False)
            train_losses.append(train_mse)
            val_losses.append(val_mse)
            epochs_done = epoch

            if val_mse < best_val:
                best_val = val_mse
                patience_counter = 0
                torch.save(model.state_dict(), self._ckpt)
            else:
                patience_counter += 1
                if patience_counter >= self._cfg.early_stopping_patience:
                    break

        self._save_log(train_losses, val_losses)
        return TrainingResult(
            best_val_mse=best_val,
            epochs_trained=epochs_done,
            stopped_early=epochs_done < self._cfg.max_epochs,
            checkpoint_path=str(self._ckpt),
            train_losses=train_losses,
            val_losses=val_losses,
        )

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device,
        training: bool,
    ) -> float:
        """Run one epoch; return mean MSE over all batches."""
        model.train(training)
        total, n = 0.0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = self._criterion(pred, yb)
                if training and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    if self._cfg.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), self._cfg.gradient_clip_norm
                        )
                    optimizer.step()
                total += loss.item() * len(xb)
                n += len(xb)
        return total / n if n > 0 else 0.0

    def _save_log(self, train_losses: list[float], val_losses: list[float]) -> None:
        self._log.parent.mkdir(parents=True, exist_ok=True)
        with self._log.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "train_mse", "val_mse"])
            for i, (t, v) in enumerate(zip(train_losses, val_losses, strict=True), 1):
                writer.writerow([i, t, v])
