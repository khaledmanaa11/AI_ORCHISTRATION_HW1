"""DatasetSDK — single public entry point for all dataset generation logic.

All external consumers call only this class. No business logic lives here;
the SDK orchestrates the service layer and returns a DatasetResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from signal_dataset.services.dataset_builder import DatasetBuilder
from signal_dataset.services.noise_injector import NoiseInjector
from signal_dataset.services.signal_generator import SignalGenerator
from signal_dataset.services.windower import Windower
from signal_dataset.shared.config import ConfigManager
from signal_dataset.shared.gatekeeper import ApiGatekeeper, RateLimitConfig
from signal_dataset.shared.version import __version__


@dataclass
class DatasetResult:
    """Result returned by DatasetSDK.generate_dataset()."""

    dataset_path: Path
    raw_path: Path
    split_result: object  # SplitResult from dataset_builder


class DatasetSDK:
    """Single public API for generating the noisy sine signal dataset.

    Flow: ConfigManager → SignalGenerator → NoiseInjector → Windower → DatasetBuilder
    """

    def __init__(
        self,
        config_path: str | Path = "config/setup.json",
        rate_limits_path: str | Path = "config/rate_limits.json",
    ) -> None:
        """Load and validate configuration from the given paths."""
        self._config_mgr = ConfigManager(Path(config_path), Path(rate_limits_path))
        self._cfg = self._config_mgr.load()

        rl = self._config_mgr.rate_limits["rate_limits"]["services"]["default"]
        rate_cfg = RateLimitConfig(
            requests_per_minute=rl["requests_per_minute"],
            requests_per_hour=rl["requests_per_hour"],
            concurrent_max=rl["concurrent_max"],
            retry_after_seconds=rl["retry_after_seconds"],
            max_retries=rl["max_retries"],
            queue_max_depth=rl["queue_max_depth"],
        )
        self._gatekeeper = ApiGatekeeper(rate_cfg)

    def generate_dataset(self) -> DatasetResult:
        """Run the full pipeline and save dataset to disk.

        Returns DatasetResult with paths to the saved .npz files.
        """
        cfg = self._cfg

        gen = SignalGenerator(cfg.dataset, cfg.signals)
        t = gen.generate_time_axis()
        clean_signals = gen.generate_all()

        injector = NoiseInjector(cfg.noise, cfg.dataset, cfg.dataset.random_seed)
        noisy_signals = injector.inject_all(clean_signals, cfg.signals, t)

        windower = Windower(cfg.dataset.window_size, 5, cfg.dataset.random_seed)
        window_result = windower.build_windows(noisy_signals, clean_signals)

        builder = DatasetBuilder(cfg.dataset, cfg.output)
        split = builder.split(window_result.X, window_result.C, window_result.y)
        dataset_path = builder.save_dataset(split)
        raw_path = builder.save_raw_signals(clean_signals, noisy_signals, t)

        return DatasetResult(dataset_path=dataset_path, raw_path=raw_path, split_result=split)

    def get_version(self) -> str:
        """Return the package version string."""
        return __version__
