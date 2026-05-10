"""Configuration loader and validator for the signal_dataset package.

Loads setup.json and rate_limits.json, parses them into typed dataclasses,
and raises ConfigValidationError when values violate domain constraints.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigValidationError(Exception):
    """Raised when a config value fails domain validation."""


@dataclass
class SignalConfig:
    """Parameters for a single sine wave component."""

    id: str
    amplitude: float
    frequency_hz: float
    phase_rad: float


@dataclass
class GaussianNoiseConfig:
    """Per-signal Gaussian noise standard deviations."""

    sigma_amp: list[float]
    sigma_phase: list[float]


@dataclass
class BurstNoiseConfig:
    """Parameters controlling random burst noise events."""

    probability: float
    duration_range_samples: list[int]
    amp_magnitude: float
    phase_magnitude: float


@dataclass
class NoiseConfig:
    """Combined noise configuration (Gaussian + burst)."""

    gaussian: GaussianNoiseConfig
    burst: BurstNoiseConfig


@dataclass
class DatasetConfig:
    """Sampling and split parameters for the dataset."""

    duration_sec: float
    sample_rate_hz: float
    window_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int


@dataclass
class OutputConfig:
    """File output paths for the generated dataset."""

    dataset_path: str
    raw_signals_path: str
    results_dir: str


@dataclass
class AppConfig:
    """Top-level application configuration."""

    dataset: DatasetConfig
    signals: list[SignalConfig]
    noise: NoiseConfig
    output: OutputConfig


class ConfigManager:
    """Loads, parses, and validates setup.json and rate_limits.json.

    Usage:
        mgr = ConfigManager(config_path, rate_limits_path)
        app_cfg = mgr.load()
    """

    def __init__(self, config_path: str | Path, rate_limits_path: str | Path) -> None:
        """Initialise with paths to both config files."""
        self._config_path = Path(config_path)
        self._rate_limits_path = Path(rate_limits_path)
        self.rate_limits: dict[str, Any] | None = None

    def load(self) -> AppConfig:
        """Load and validate both config files; return AppConfig."""
        raw = self._load_json(self._config_path)
        self.rate_limits = self._load_json(self._rate_limits_path)
        config = AppConfig(
            dataset=self._parse_dataset(raw["dataset"]),
            signals=self._parse_signals(raw["signals"]),
            noise=self._parse_noise(raw["noise"]),
            output=self._parse_output(raw.get("output", {})),
        )
        self._validate(config)
        return config

    def _load_json(self, path: Path) -> dict[str, Any]:
        """Read and parse a JSON file; raise FileNotFoundError or JSONDecodeError on failure."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            return json.load(f)

    def _parse_dataset(self, raw: dict) -> DatasetConfig:
        """Parse the dataset block into a DatasetConfig dataclass."""
        return DatasetConfig(
            duration_sec=raw["duration_sec"],
            sample_rate_hz=raw["sample_rate_hz"],
            window_size=raw["window_size"],
            train_ratio=raw["train_ratio"],
            val_ratio=raw["val_ratio"],
            test_ratio=raw["test_ratio"],
            random_seed=raw["random_seed"],
        )

    def _parse_signals(self, raw: list) -> list[SignalConfig]:
        """Parse the signals array into a list of SignalConfig dataclasses."""
        return [
            SignalConfig(
                id=s["id"],
                amplitude=s["amplitude"],
                frequency_hz=s["frequency_hz"],
                phase_rad=s["phase_rad"],
            )
            for s in raw
        ]

    def _parse_noise(self, raw: dict) -> NoiseConfig:
        """Parse the noise block into a NoiseConfig dataclass."""
        g = raw["gaussian"]
        b = raw["burst"]
        return NoiseConfig(
            gaussian=GaussianNoiseConfig(
                sigma_amp=g["sigma_amp"],
                sigma_phase=g["sigma_phase"],
            ),
            burst=BurstNoiseConfig(
                probability=b["probability"],
                duration_range_samples=b["duration_range_samples"],
                amp_magnitude=b["amp_magnitude"],
                phase_magnitude=b["phase_magnitude"],
            ),
        )

    def _parse_output(self, raw: dict) -> OutputConfig:
        """Parse the output block into an OutputConfig dataclass."""
        return OutputConfig(
            dataset_path=raw.get("dataset_path", "data/dataset.npz"),
            raw_signals_path=raw.get("raw_signals_path", "data/signals_raw.npz"),
            results_dir=raw.get("results_dir", "results/"),
        )

    def _validate(self, cfg: AppConfig) -> None:
        """Raise ConfigValidationError if any field violates domain constraints."""
        d = cfg.dataset
        if d.duration_sec <= 0:
            raise ConfigValidationError(f"duration_sec must be > 0, got {d.duration_sec}")
        if d.sample_rate_hz <= 0:
            raise ConfigValidationError(f"sample_rate_hz must be > 0, got {d.sample_rate_hz}")
        if d.window_size <= 0:
            raise ConfigValidationError(f"window_size must be > 0, got {d.window_size}")
        ratio_sum = d.train_ratio + d.val_ratio + d.test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ConfigValidationError(f"split ratios must sum to 1.0, got {ratio_sum}")
        for r in (d.train_ratio, d.val_ratio, d.test_ratio):
            if not (0 < r < 1):
                raise ConfigValidationError(f"each split ratio must be in (0,1), got {r}")
        if not cfg.signals:
            raise ConfigValidationError("signals list must not be empty")
        for s in cfg.signals:
            if s.amplitude <= 0:
                raise ConfigValidationError(f"signal {s.id} amplitude must be > 0, got {s.amplitude}")
            if s.frequency_hz <= 0:
                raise ConfigValidationError(f"signal {s.id} frequency_hz must be > 0, got {s.frequency_hz}")
        g = cfg.noise.gaussian
        if any(v < 0 for v in g.sigma_amp):
            raise ConfigValidationError(f"sigma_amp values must be >= 0, got {g.sigma_amp}")
        if any(v < 0 for v in g.sigma_phase):
            raise ConfigValidationError(f"sigma_phase values must be >= 0, got {g.sigma_phase}")
        b = cfg.noise.burst
        if not (0 <= b.probability <= 1):
            raise ConfigValidationError(f"burst probability must be in [0,1], got {b.probability}")
        lo, hi = b.duration_range_samples
        if lo >= hi:
            raise ConfigValidationError(f"burst duration_range_samples[0] must be < [1], got {lo},{hi}")
        if len(g.sigma_amp) != len(cfg.signals):
            raise ConfigValidationError("len(sigma_amp) must equal len(signals)")
