"""DataLoaderService — loads dataset.npz and builds PyTorch DataLoaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_signal import constants as const


@dataclass
class DataBundle:
    """Container for all DataLoaders and raw numpy arrays from one dataset split."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    seq_train_loader: DataLoader
    seq_val_loader: DataLoader
    seq_test_loader: DataLoader
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    C_train: np.ndarray
    C_val: np.ndarray
    C_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


_REQUIRED_KEYS = [
    const.DATASET_KEY_X_TRAIN, const.DATASET_KEY_X_VAL, const.DATASET_KEY_X_TEST,
    const.DATASET_KEY_C_TRAIN, const.DATASET_KEY_C_VAL, const.DATASET_KEY_C_TEST,
    const.DATASET_KEY_Y_TRAIN, const.DATASET_KEY_Y_VAL, const.DATASET_KEY_Y_TEST,
]


class DataLoaderService:
    """Loads dataset.npz and exposes FCN and RNN/LSTM DataLoaders.

    FCN input: concat([X_window, C]) → shape (N, 15)
    RNN/LSTM input (broadcast strategy): concat window + C per step → shape (N, window_size, 6)
    """

    def __init__(
        self,
        dataset_path: Path,
        batch_size: int,
        seed: int = 42,
        c_injection_strategy: str = "broadcast",
    ) -> None:
        """Initialise with dataset path, batch size, RNG seed, and C injection strategy."""
        self._path = Path(dataset_path)
        self._batch_size = batch_size
        self._seed = seed
        self._c_strategy = c_injection_strategy

    def load(self) -> DataBundle:
        """Load .npz, build tensors, and return a DataBundle with all loaders."""
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._path}")

        data = np.load(self._path)
        self._validate_keys(data)

        x_tr = data[const.DATASET_KEY_X_TRAIN].astype(np.float32)
        x_va = data[const.DATASET_KEY_X_VAL].astype(np.float32)
        x_te = data[const.DATASET_KEY_X_TEST].astype(np.float32)
        c_tr = data[const.DATASET_KEY_C_TRAIN].astype(np.float32)
        c_va = data[const.DATASET_KEY_C_VAL].astype(np.float32)
        c_te = data[const.DATASET_KEY_C_TEST].astype(np.float32)
        y_tr = data[const.DATASET_KEY_Y_TRAIN].astype(np.float32).reshape(-1, 1)
        y_va = data[const.DATASET_KEY_Y_VAL].astype(np.float32).reshape(-1, 1)
        y_te = data[const.DATASET_KEY_Y_TEST].astype(np.float32).reshape(-1, 1)

        return DataBundle(
            train_loader=self._make_loader(
                self._fcn_tensor(x_tr, c_tr), self._to_tensor(y_tr), shuffle=True
            ),
            val_loader=self._make_loader(
                self._fcn_tensor(x_va, c_va), self._to_tensor(y_va), shuffle=False
            ),
            test_loader=self._make_loader(
                self._fcn_tensor(x_te, c_te), self._to_tensor(y_te), shuffle=False
            ),
            seq_train_loader=self._make_loader(
                self._seq_tensor(x_tr, c_tr), self._to_tensor(y_tr), shuffle=True
            ),
            seq_val_loader=self._make_loader(
                self._seq_tensor(x_va, c_va), self._to_tensor(y_va), shuffle=False
            ),
            seq_test_loader=self._make_loader(
                self._seq_tensor(x_te, c_te), self._to_tensor(y_te), shuffle=False
            ),
            X_train=x_tr, X_val=x_va, X_test=x_te,
            C_train=c_tr, C_val=c_va, C_test=c_te,
            y_train=y_tr, y_val=y_va, y_test=y_te,
        )

    def _validate_keys(self, data: np.lib.npyio.NpzFile) -> None:
        """Raise KeyError if any required npz key is missing."""
        for key in _REQUIRED_KEYS:
            if key not in data:
                raise KeyError(f"Missing key in dataset.npz: '{key}'")

    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr)

    @staticmethod
    def _fcn_tensor(x_win: np.ndarray, cv: np.ndarray) -> torch.Tensor:
        """Concatenate window and selector along feature axis for FCN input."""
        return torch.from_numpy(np.concatenate([x_win, cv], axis=1))

    def _seq_tensor(self, x_win: np.ndarray, c: np.ndarray) -> torch.Tensor:
        """Build RNN/LSTM input by broadcasting C selector across all timesteps.

        broadcast strategy: concat window(1) + C(5) per step → (N, seq_len, 6)
        """
        n, w = x_win.shape
        x_3d = x_win.reshape(n, w, 1)
        c_3d = np.broadcast_to(c[:, np.newaxis, :], (n, w, c.shape[1])).copy()
        return torch.from_numpy(np.concatenate([x_3d, c_3d], axis=2))

    def _make_loader(
        self, x: torch.Tensor, y: torch.Tensor, shuffle: bool
    ) -> DataLoader:
        ds = TensorDataset(x, y)
        generator = torch.Generator().manual_seed(self._seed) if shuffle else None
        return DataLoader(
            ds, batch_size=self._batch_size, shuffle=shuffle, generator=generator
        )
