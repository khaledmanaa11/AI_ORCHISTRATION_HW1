"""Dataclass definitions for all configuration sections."""

from __future__ import annotations

from dataclasses import dataclass, field


class ConfigValidationError(Exception):
    """Raised when a config value fails validation."""


@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    val_split: float
    random_seed: int
    device: str = "cpu"
    gradient_clip_norm: float | None = None


@dataclass
class FCNConfig:
    """Architecture config for the Fully-Connected Network."""

    hidden_sizes: list[int]
    dropout_rate: float
    output_size: int


@dataclass
class RNNConfig:
    """Architecture config for the Vanilla RNN."""

    hidden_size: int
    nonlinearity: str
    num_layers: int
    output_size: int


@dataclass
class LSTMConfig:
    """Architecture config for the LSTM."""

    hidden_size: int
    num_layers: int
    dense_hidden_size: int
    output_size: int


@dataclass
class OutputConfig:
    """Paths for saving results, checkpoints, and plots."""

    results_dir: str
    checkpoint_dir: str
    fcn_checkpoint: str
    rnn_checkpoint: str
    lstm_checkpoint: str
    comparison_table_path: str
    scaler_params_path: str
    training_log_fcn: str = field(default="results/training_log_fcn.csv")
    training_log_rnn: str = field(default="results/training_log_rnn.csv")
    training_log_lstm: str = field(default="results/training_log_lstm.csv")


@dataclass
class AppConfig:
    """Top-level application configuration."""

    training: TrainingConfig
    fcn: FCNConfig
    rnn: RNNConfig
    lstm: LSTMConfig
    output: OutputConfig
