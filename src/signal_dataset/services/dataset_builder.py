"""Dataset assembly, chronological splitting, and .npz serialisation.

Takes the windowed arrays (X, C, y) from Windower and:
1. Splits them chronologically into train/val/test without shuffling.
2. Saves dataset.npz with all 9 arrays (X, C, y per split).
3. Saves signals_raw.npz with stacked clean and noisy signal matrices.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from signal_dataset.constants import (
    DATASET_NPZ_KEY_C_TEST,
    DATASET_NPZ_KEY_C_TRAIN,
    DATASET_NPZ_KEY_C_VAL,
    DATASET_NPZ_KEY_X_TEST,
    DATASET_NPZ_KEY_X_TRAIN,
    DATASET_NPZ_KEY_X_VAL,
    DATASET_NPZ_KEY_Y_TEST,
    DATASET_NPZ_KEY_Y_TRAIN,
    DATASET_NPZ_KEY_Y_VAL,
    RAW_NPZ_KEY_CLEAN,
    RAW_NPZ_KEY_NOISY,
    RAW_NPZ_KEY_TIME,
    SIGNAL_IDS,
)
from signal_dataset.shared.config import DatasetConfig, OutputConfig


@dataclass
class SplitResult:
    """Holds all 9 arrays produced by a chronological train/val/test split."""

    X_train: np.ndarray
    C_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    C_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    C_test: np.ndarray
    y_test: np.ndarray


class DatasetBuilder:
    """Assembles, splits, and saves the dataset to .npz files.

    Input:  DatasetConfig (split ratios), OutputConfig (file paths)
    Output: data/dataset.npz and data/signals_raw.npz
    """

    def __init__(self, cfg: DatasetConfig, output_cfg: OutputConfig) -> None:
        """Store config references; derive split fractions from cfg."""
        self._cfg = cfg
        self._output_cfg = output_cfg

    def split(self, X: np.ndarray, C: np.ndarray, y: np.ndarray) -> SplitResult:  # noqa: N803
        """Chronologically split X, C, y into train/val/test.

        No shuffling — time ordering is preserved across all three arrays.
        """
        n = X.shape[0]  # noqa: N806
        n_train = int(n * self._cfg.train_ratio)
        n_val = int(n * self._cfg.val_ratio)
        return SplitResult(
            X_train=X[:n_train], C_train=C[:n_train], y_train=y[:n_train],  # noqa: N806
            X_val=X[n_train:n_train + n_val], C_val=C[n_train:n_train + n_val], y_val=y[n_train:n_train + n_val],  # noqa: N806
            X_test=X[n_train + n_val:], C_test=C[n_train + n_val:], y_test=y[n_train + n_val:],  # noqa: N806
        )

    def save_dataset(self, split: SplitResult) -> Path:
        """Save all 9 split arrays to a compressed .npz file.

        Returns the Path to the saved file.
        """
        path = Path(self._output_cfg.dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{
                DATASET_NPZ_KEY_X_TRAIN: split.X_train,
                DATASET_NPZ_KEY_C_TRAIN: split.C_train,
                DATASET_NPZ_KEY_Y_TRAIN: split.y_train,
                DATASET_NPZ_KEY_X_VAL: split.X_val,
                DATASET_NPZ_KEY_C_VAL: split.C_val,
                DATASET_NPZ_KEY_Y_VAL: split.y_val,
                DATASET_NPZ_KEY_X_TEST: split.X_test,
                DATASET_NPZ_KEY_C_TEST: split.C_test,
                DATASET_NPZ_KEY_Y_TEST: split.y_test,
            },
        )
        return path

    def save_raw_signals(
        self, clean: dict[str, np.ndarray], noisy: dict[str, np.ndarray], t: np.ndarray
    ) -> Path:
        """Stack all 5 clean and noisy signals into (5, N) matrices and save.

        Row order follows SIGNAL_IDS: s1, s2, s3, s4, s5.
        Returns the Path to the saved file.
        """
        path = Path(self._output_cfg.raw_signals_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        clean_mat = np.vstack([clean[sid] for sid in SIGNAL_IDS])
        noisy_mat = np.vstack([noisy[sid] for sid in SIGNAL_IDS])
        np.savez_compressed(
            path,
            **{RAW_NPZ_KEY_CLEAN: clean_mat, RAW_NPZ_KEY_NOISY: noisy_mat, RAW_NPZ_KEY_TIME: t},
        )
        return path
