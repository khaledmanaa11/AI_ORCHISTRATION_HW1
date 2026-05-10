"""Gaussian + burst noise injection service.

Applies amplitude and phase noise to clean sine signals using the formula:
    s_noisy(t) = (A + N_amp(t)) * sin(2π*f*t + φ + N_phase(t))

N_amp and N_phase each combine Gaussian white noise with random burst events.
All parameters come from NoiseConfig — no hardcoded values.
"""

import numpy as np

from signal_dataset.constants import SIGNAL_IDS, TWO_PI
from signal_dataset.shared.config import DatasetConfig, NoiseConfig, SignalConfig


class NoiseInjector:
    """Injects configurable amplitude and phase noise into clean sine signals.

    Input:  NoiseConfig, DatasetConfig, random seed
    Output: Dict[str, np.ndarray] with keys s1..s5, each shape (n_samples,)
    """

    def __init__(self, noise_cfg: NoiseConfig, dataset_cfg: DatasetConfig, seed: int) -> None:
        """Initialise with noise params and a seeded random number generator."""
        self._noise_cfg = noise_cfg
        self._dataset_cfg = dataset_cfg
        self._rng = np.random.default_rng(seed)

    def _generate_gaussian_noise(self, sigma: float, n: int) -> np.ndarray:
        """Draw n samples from N(0, sigma²). Returns zeros if sigma == 0."""
        if sigma == 0:
            return np.zeros(n)
        return self._rng.normal(0.0, sigma, n)

    def _generate_burst_mask(self, n: int) -> np.ndarray:
        """Generate a binary burst mask of length n.

        Each sample independently starts a burst with the configured probability.
        Burst runs are extended to a uniformly sampled duration in [d_min, d_max].
        """
        b = self._noise_cfg.burst
        mask = np.zeros(n, dtype=float)
        if b.probability == 0.0:
            return mask
        i = 0
        while i < n:
            if self._rng.random() < b.probability:
                dur = int(self._rng.integers(b.duration_range_samples[0], b.duration_range_samples[1] + 1))
                end = min(i + dur, n)
                mask[i:end] = 1.0
                i = end
            else:
                i += 1
        return mask

    def _generate_burst_noise(self, mask: np.ndarray, magnitude: float) -> np.ndarray:
        """Uniform random values in [-magnitude, +magnitude] where mask==1, else 0."""
        noise = np.zeros(len(mask))
        active = mask == 1
        if np.any(active):
            n_active = int(np.sum(active))
            noise[active] = self._rng.uniform(-magnitude, magnitude, n_active)
        return noise

    def _compute_n_amp(self, signal_idx: int, n: int) -> np.ndarray:
        """Compute N_amp = gaussian_amp + burst_amp for one signal."""
        g = self._noise_cfg.gaussian
        b = self._noise_cfg.burst
        gauss = self._generate_gaussian_noise(g.sigma_amp[signal_idx], n)
        mask = self._generate_burst_mask(n)
        burst = self._generate_burst_noise(mask, b.amp_magnitude)
        return gauss + burst

    def _compute_n_phase(self, signal_idx: int, n: int) -> np.ndarray:
        """Compute N_phase = gaussian_phase + burst_phase for one signal."""
        g = self._noise_cfg.gaussian
        b = self._noise_cfg.burst
        gauss = self._generate_gaussian_noise(g.sigma_phase[signal_idx], n)
        mask = self._generate_burst_mask(n)
        burst = self._generate_burst_noise(mask, b.phase_magnitude)
        return gauss + burst

    def inject_single(
        self, clean: np.ndarray, signal_cfg: SignalConfig, signal_idx: int, t: np.ndarray
    ) -> np.ndarray:
        """Apply amplitude and phase noise to one clean signal.

        Formula: (A + N_amp) * sin(2π*f*t + φ + N_phase)
        """
        n = len(t)
        n_amp = self._compute_n_amp(signal_idx, n)
        n_phase = self._compute_n_phase(signal_idx, n)
        return (signal_cfg.amplitude + n_amp) * np.sin(
            TWO_PI * signal_cfg.frequency_hz * t + signal_cfg.phase_rad + n_phase
        )

    def inject_all(
        self, clean_signals: dict, signal_configs: list[SignalConfig], t: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Inject noise into all 4 base signals; compute noisy composite s5.

        s5_noisy = s1_noisy + s2_noisy + s3_noisy + s4_noisy
        """
        noisy: dict[str, np.ndarray] = {}
        base_ids = [sid for sid in SIGNAL_IDS if sid != "s5"]
        for idx, sid in enumerate(base_ids):
            cfg = signal_configs[idx]
            noisy[sid] = self.inject_single(clean_signals[sid], cfg, idx, t)
        noisy["s5"] = sum(noisy[sid] for sid in base_ids)
        return noisy
