"""ConfigManager — loads, parses, and validates config/setup.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neural_signal.shared.config_types import (
    AppConfig,
    ConfigValidationError,
    FCNConfig,
    LSTMConfig,
    OutputConfig,
    RNNConfig,
    TrainingConfig,
)

# Re-export so existing `from neural_signal.shared.config import X` still works.
__all__ = [
    "AppConfig", "ConfigValidationError", "ConfigManager",
    "FCNConfig", "LSTMConfig", "OutputConfig", "RNNConfig", "TrainingConfig",
]


class ConfigManager:
    """Loads config/setup.json and rate_limits.json into validated dataclasses."""

    def __init__(self, config_path: Path, rate_limits_path: Path) -> None:
        """Initialise with paths to both config files."""
        self._config_path = Path(config_path)
        self._rate_limits_path = Path(rate_limits_path)

    def load(self) -> AppConfig:
        """Parse both JSON files and return a validated AppConfig."""
        raw = self._load_json(self._config_path)
        cfg = AppConfig(
            training=self._parse_training(raw["training"]),
            fcn=self._parse_fcn(raw["fcn"]),
            rnn=self._parse_rnn(raw["rnn"]),
            lstm=self._parse_lstm(raw["lstm"]),
            output=self._parse_output(raw["output"]),
        )
        self._validate(cfg)
        return cfg

    def _load_json(self, path: Path) -> dict[str, Any]:
        """Read and decode a JSON file, raising on missing file or bad syntax."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open() as fh:
            return json.load(fh)

    def _parse_training(self, raw: dict) -> TrainingConfig:
        return TrainingConfig(
            batch_size=raw["batch_size"],
            learning_rate=raw["learning_rate"],
            weight_decay=raw["weight_decay"],
            max_epochs=raw["max_epochs"],
            early_stopping_patience=raw["early_stopping_patience"],
            val_split=raw["val_split"],
            random_seed=raw["random_seed"],
            device=raw.get("device", "cpu"),
            gradient_clip_norm=raw.get("gradient_clip_norm"),
        )

    def _parse_fcn(self, raw: dict) -> FCNConfig:
        return FCNConfig(
            hidden_sizes=raw["hidden_sizes"],
            dropout_rate=raw["dropout_rate"],
            output_size=raw["output_size"],
        )

    def _parse_rnn(self, raw: dict) -> RNNConfig:
        return RNNConfig(
            hidden_size=raw["hidden_size"],
            nonlinearity=raw["nonlinearity"],
            num_layers=raw["num_layers"],
            output_size=raw["output_size"],
        )

    def _parse_lstm(self, raw: dict) -> LSTMConfig:
        return LSTMConfig(
            hidden_size=raw["hidden_size"],
            num_layers=raw["num_layers"],
            dense_hidden_size=raw["dense_hidden_size"],
            output_size=raw["output_size"],
        )

    def _parse_output(self, raw: dict) -> OutputConfig:
        rd = raw.get("results_dir", "results/")
        return OutputConfig(
            results_dir=rd,
            checkpoint_dir=raw["checkpoint_dir"],
            fcn_checkpoint=raw["fcn_checkpoint"],
            rnn_checkpoint=raw["rnn_checkpoint"],
            lstm_checkpoint=raw["lstm_checkpoint"],
            comparison_table_path=raw["comparison_table_path"],
            scaler_params_path=raw["scaler_params_path"],
            training_log_fcn=raw.get("training_log_fcn", rd + "training_log_fcn.csv"),
            training_log_rnn=raw.get("training_log_rnn", rd + "training_log_rnn.csv"),
            training_log_lstm=raw.get("training_log_lstm", rd + "training_log_lstm.csv"),
        )

    def _validate(self, cfg: AppConfig) -> None:
        """Raise ConfigValidationError for any invalid field."""
        t = cfg.training
        self._check(t.learning_rate > 0, "learning_rate", t.learning_rate, "> 0")
        self._check(t.batch_size >= 1, "batch_size", t.batch_size, ">= 1")
        self._check(t.max_epochs >= 1, "max_epochs", t.max_epochs, ">= 1")
        self._check(t.early_stopping_patience >= 1, "early_stopping_patience",
                    t.early_stopping_patience, ">= 1")
        self._check(0.0 < t.val_split < 1.0, "val_split", t.val_split, "in (0, 1)")
        f = cfg.fcn
        self._check(0.0 <= f.dropout_rate < 1.0, "dropout_rate", f.dropout_rate, "in [0, 1)")
        self._check(len(f.hidden_sizes) > 0, "fcn.hidden_sizes", f.hidden_sizes, "non-empty")
        self._check(cfg.rnn.hidden_size > 0, "rnn.hidden_size", cfg.rnn.hidden_size, "> 0")
        self._check(cfg.lstm.hidden_size > 0, "lstm.hidden_size", cfg.lstm.hidden_size, "> 0")

    @staticmethod
    def _check(condition: bool, field: str, value: Any, constraint: str) -> None:
        if not condition:
            raise ConfigValidationError(
                f"Invalid config: '{field}' = {value!r} must be {constraint}"
            )
