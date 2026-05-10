"""Clean sine wave generation service.

Generates individual sine components (s1–s4) and their composite (s5)
using parameters loaded from configuration. No magic numbers — all values
come from SignalConfig and DatasetConfig.
"""

import numpy as np

from signal_dataset.constants import SIGNAL_IDS, TWO_PI
from signal_dataset.shared.config import DatasetConfig, SignalConfig


class SignalGenerator:
    """Generates clean sine wave signals from configuration.

    Input:  DatasetConfig (sampling params), list[SignalConfig] (per-signal params)
    Output: Dict[str, np.ndarray] with keys s1..s5, each shape (n_samples,)
    """

    def __init__(self, dataset_cfg: DatasetConfig, signals: list[SignalConfig]) -> None:
        """Store config; compute derived sampling constants at init time."""
        self._dataset_cfg = dataset_cfg
        self._signals = signals

    def generate_time_axis(self) -> np.ndarray:
        """Return the time axis as a 1-D array with endpoint=False.

        Uses linspace so the step equals exactly 1/sample_rate_hz.
        """
        cfg = self._dataset_cfg
        n_samples = int(cfg.duration_sec * cfg.sample_rate_hz)
        return np.linspace(0, cfg.duration_sec, n_samples, endpoint=False)

    def generate_single(self, signal_cfg: SignalConfig, t: np.ndarray) -> np.ndarray:
        """Evaluate a single sine component over the given time axis.

        Formula: A * sin(2π * f * t + φ)
        """
        return signal_cfg.amplitude * np.sin(TWO_PI * signal_cfg.frequency_hz * t + signal_cfg.phase_rad)

    def generate_composite(self, components: dict[str, np.ndarray]) -> np.ndarray:
        """Sum all base components (s1+s2+s3+s4) to produce the composite signal s5."""
        base_ids = [sid for sid in SIGNAL_IDS if sid != "s5"]
        return sum(components[sid] for sid in base_ids)

    def generate_all(self) -> dict[str, np.ndarray]:
        """Generate all 5 signals and return them keyed by signal ID.

        Returns:
            Dict with keys 's1'..'s5'; values are shape (n_samples,) float64 arrays.
        """
        t = self.generate_time_axis()
        signals: dict[str, np.ndarray] = {}
        for sig_cfg in self._signals:
            signals[sig_cfg.id] = self.generate_single(sig_cfg, t)
        signals["s5"] = self.generate_composite(signals)
        return signals
