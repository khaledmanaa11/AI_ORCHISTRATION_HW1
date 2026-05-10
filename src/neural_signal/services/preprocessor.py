"""Preprocessor — z-score normalisation fitted on training data only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ScalerParams:
    """Fitted mean and std for z-score normalisation."""

    mean: np.ndarray
    std: np.ndarray


class Preprocessor:
    """Z-score normaliser that fits on train set and applies to val/test.

    Prevents data leakage: mean and std are computed from training data only.
    Zero-std columns are replaced with 1 to avoid division errors.
    """

    def __init__(self, scaler_path: Path) -> None:
        """Initialise with output path for persisting scaler params."""
        self._scaler_path = Path(scaler_path)
        self._params: ScalerParams | None = None

    def fit(self, x_train: np.ndarray) -> ScalerParams:
        """Compute mean and std from x_train and store fitted params."""
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std = np.where(std == 0, 1.0, std)  # avoid division by zero
        self._params = ScalerParams(mean=mean, std=std)
        return self._params

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply (x - mean) / std using params fitted on training data."""
        if self._params is None:
            raise RuntimeError("Preprocessor.fit() must be called before transform().")
        return (x - self._params.mean) / self._params.std

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform x_train in one call."""
        self.fit(x_train)
        return self.transform(x_train)

    def save_params(self) -> None:
        """Persist mean and std to JSON for reproducibility."""
        if self._params is None:
            raise RuntimeError("No params to save — call fit() first.")
        self._scaler_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self._params.mean.tolist(),
            "std": self._params.std.tolist(),
        }
        self._scaler_path.write_text(json.dumps(payload, indent=2))

    def load_params(self) -> None:
        """Load mean and std from a previously saved JSON file."""
        raw = json.loads(self._scaler_path.read_text())
        self._params = ScalerParams(
            mean=np.array(raw["mean"], dtype=np.float32),
            std=np.array(raw["std"], dtype=np.float32),
        )
