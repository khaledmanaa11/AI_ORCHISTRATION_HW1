"""Sliding window construction with one-hot signal selector.

Extracts windows from the composite noisy signal (s5_noisy only),
assigns a random one-hot selector C for each window, and records
the clean scalar label from the C-selected signal at the window center.
"""

from dataclasses import dataclass

import numpy as np

from signal_dataset.constants import SIGNAL_IDS


@dataclass
class WindowResult:
    """Output of Windower.build_windows().

    X: (N, W)  — composite noisy windows sourced from s5_noisy
    C: (N, 5)  — one-hot signal selectors (length == n_signals, NOT window size)
    y: (N, 1)  — scalar clean label for the C-selected signal at window center
    """

    X: np.ndarray  # noqa: N815
    C: np.ndarray  # noqa: N815
    y: np.ndarray


class Windower:
    """Builds sliding-window dataset samples from noisy and clean signals.

    Input:  noisy_signals dict, clean_signals dict, window_size W, n_signals, seed
    Output: WindowResult with X (N,W), C (N,5), y (N,1)
    """

    def __init__(self, window_size: int, n_signals: int, seed: int) -> None:
        """Initialise with window size, number of signal classes, and RNG seed."""
        self._W = window_size  # noqa: N815
        self._n_signals = n_signals
        self._rng = np.random.default_rng(seed)

    def _generate_one_hot(self) -> np.ndarray:
        """Return a one-hot vector of length n_signals with a random active index."""
        vec = np.zeros(self._n_signals, dtype=float)
        idx = int(self._rng.integers(0, self._n_signals))
        vec[idx] = 1.0
        return vec

    def _build_features(self, src: np.ndarray, n: int) -> np.ndarray:
        """Slice n windows of size W from src into an (N, W) array."""
        w = self._W
        x_arr = np.empty((n, w), dtype=float)
        for k in range(n):
            x_arr[k] = src[k : k + w]
        return x_arr

    def _build_labels(self, clean_signals: dict, c_arr: np.ndarray, n: int) -> np.ndarray:
        """Extract scalar clean label for each window using the C-selected signal."""
        y = np.empty((n, 1), dtype=float)
        half = self._W // 2  # noqa: N806
        for k in range(n):
            i = int(np.argmax(c_arr[k]))
            sid = SIGNAL_IDS[i]
            y[k, 0] = clean_signals[sid][k + half]
        return y

    def build_windows(
        self, noisy_signals: dict[str, np.ndarray], clean_signals: dict[str, np.ndarray]
    ) -> WindowResult:
        """Build the full (X, C, y) arrays from noisy and clean signal dicts.

        Source signal for X is always s5_noisy (composite noisy).
        Raises ValueError if s5 is missing from noisy_signals or signal is too short.
        """
        if "s5" not in noisy_signals:
            raise ValueError("noisy_signals must contain key 's5' (composite noisy signal)")
        src = noisy_signals["s5"]
        if len(src) < self._W:  # noqa: N806
            raise ValueError(f"Signal length {len(src)} is shorter than window_size {self._W}")  # noqa: N806

        n = len(src) - self._W + 1  # noqa: N806
        x_arr = self._build_features(src, n)
        c_arr = np.vstack([self._generate_one_hot() for _ in range(n)])
        y = self._build_labels(clean_signals, c_arr, n)
        return WindowResult(X=x_arr, C=c_arr, y=y)
