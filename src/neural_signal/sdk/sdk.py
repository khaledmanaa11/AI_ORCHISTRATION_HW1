"""NeuralSignalSDK — single public entry point for the entire pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from neural_signal import constants as const
from neural_signal.models.fcn import FCNModel
from neural_signal.models.lstm import LSTMModel
from neural_signal.models.rnn import RNNModel
from neural_signal.sdk._types import RunResult
from neural_signal.sdk._viz import VizMixin
from neural_signal.services.data_loader import DataBundle, DataLoaderService
from neural_signal.services.evaluator import EvalResult, Evaluator
from neural_signal.services.preprocessor import Preprocessor
from neural_signal.services.trainer import Trainer, TrainingResult
from neural_signal.shared.config import AppConfig, ConfigManager
from neural_signal.shared.version import __version__


class NeuralSignalSDK(VizMixin):
    """Orchestrates data loading, training, evaluation, and visualisation.

    No business logic lives here — only orchestration calls to service layer.
    Usage:
        sdk = NeuralSignalSDK(config_path, rate_limits_path)
        sdk.run_all()
    """

    def __init__(self, config_path: Path, rate_limits_path: Path) -> None:
        """Load and validate configuration from both JSON files."""
        self._cfg: AppConfig = ConfigManager(config_path, rate_limits_path).load()
        self._bundle: DataBundle | None = None
        self._train_results: dict[str, TrainingResult] = {}
        self._eval_results: dict[str, EvalResult] = {}

    def run_all(self) -> RunResult:
        """Train and evaluate all three models; save all results."""
        torch.manual_seed(self._cfg.training.random_seed)
        self._bundle = self._load_data()
        for name in const.MODEL_NAMES:
            self._train_results[name] = self.train_model(name)
        eval_dict = self.evaluate_all()
        cmp_path = Path(self._cfg.output.comparison_table_path)
        df = pd.read_csv(cmp_path) if cmp_path.exists() else pd.DataFrame()
        return RunResult(
            fcn_eval=eval_dict[const.MODEL_FCN],
            rnn_eval=eval_dict[const.MODEL_RNN],
            lstm_eval=eval_dict[const.MODEL_LSTM],
            comparison_df=df,
        )

    def get_version(self) -> str:
        """Return the current package version string."""
        return __version__

    def train_model(self, model_name: str) -> TrainingResult:
        """Train a single model by name ('fcn' | 'rnn' | 'lstm')."""
        if model_name not in const.MODEL_NAMES:
            raise ValueError(f"Unknown model '{model_name}'. Choose from {const.MODEL_NAMES}.")
        if self._bundle is None:
            self._bundle = self._load_data()
        bundle = self._bundle
        cfg = self._cfg

        model, train_loader, val_loader, ckpt, log = self._build_model_resources(
            model_name, bundle, cfg
        )
        weight_decay = cfg.training.weight_decay if model_name == const.MODEL_FCN else 0.0
        trainer = Trainer(cfg.training, Path(ckpt), Path(log))
        result = trainer.train(model, train_loader, val_loader, weight_decay=weight_decay)
        self._train_results[model_name] = result
        return result

    def evaluate_all(self) -> dict[str, EvalResult]:
        """Load best checkpoints and compute train/val/test MSE for all models."""
        if self._bundle is None:
            self._bundle = self._load_data()
        bundle = self._bundle
        cfg = self._cfg
        evaluator = Evaluator(Path(cfg.output.results_dir), device=cfg.training.device)
        results = []

        for name in const.MODEL_NAMES:
            model, train_loader, val_loader, ckpt, _ = self._build_model_resources(
                name, bundle, cfg
            )
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint for '{name}' not found: {ckpt_path}. "
                    "Run train_model() or run_all() first."
                )
            tr = self._train_results.get(name)
            test_loader = (
                bundle.test_loader if name == const.MODEL_FCN else bundle.seq_test_loader
            )
            res = evaluator.evaluate(
                model, name, ckpt_path,
                train_loader, val_loader, test_loader,
                epochs_trained=tr.epochs_trained if tr else 0,
                stopped_early=tr.stopped_early if tr else False,
            )
            self._eval_results[name] = res
            results.append(res)

        evaluator.save_comparison_table(results, Path(cfg.output.comparison_table_path))
        self._run_visualizations(bundle, results, self._build_model_resources, cfg)
        return self._eval_results

    # ------------------------------------------------------------------ helpers

    def _load_data(self) -> DataBundle:
        """Load dataset.npz, fit z-score normaliser, return DataBundle."""
        cfg = self._cfg
        bundle = DataLoaderService(
            Path("data/dataset.npz"), cfg.training.batch_size, cfg.training.random_seed
        ).load()
        pre = Preprocessor(Path(cfg.output.scaler_params_path))
        pre.fit_transform(bundle.X_train)
        pre.save_params()
        return bundle

    def _build_model_resources(
        self, name: str, bundle: DataBundle, cfg: AppConfig
    ) -> tuple:
        """Return (model, train_loader, val_loader, ckpt_path, log_path) for one model."""
        if name == const.MODEL_FCN:
            model = FCNModel(cfg.fcn)
            train_loader = bundle.train_loader
            val_loader = bundle.val_loader
            ckpt = cfg.output.fcn_checkpoint
            log = cfg.output.training_log_fcn
        elif name == const.MODEL_RNN:
            model = RNNModel(cfg.rnn)
            train_loader = bundle.seq_train_loader
            val_loader = bundle.seq_val_loader
            ckpt = cfg.output.rnn_checkpoint
            log = cfg.output.training_log_rnn
        else:
            model = LSTMModel(cfg.lstm)
            train_loader = bundle.seq_train_loader
            val_loader = bundle.seq_val_loader
            ckpt = cfg.output.lstm_checkpoint
            log = cfg.output.training_log_lstm
        return model, train_loader, val_loader, ckpt, log
