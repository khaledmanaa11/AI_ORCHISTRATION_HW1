"""NeuralSignalSDK — single public entry point for the entire pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_signal import constants as const
from neural_signal.models.fcn import FCNModel
from neural_signal.models.lstm import LSTMModel
from neural_signal.models.rnn import RNNModel
from neural_signal.services.data_loader import DataBundle, DataLoaderService
from neural_signal.services.evaluator import EvalResult, Evaluator
from neural_signal.services.preprocessor import Preprocessor
from neural_signal.services.trainer import Trainer, TrainingResult
from neural_signal.services.visualizer import Visualizer
from neural_signal.shared.config import AppConfig, ConfigManager


class NeuralSignalSDK:
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

    def run_all(self) -> dict[str, EvalResult]:
        """Train and evaluate all three models; save all results."""
        torch.manual_seed(self._cfg.training.random_seed)
        self._bundle = self._load_data()
        for name in const.MODEL_NAMES:
            self._train_results[name] = self.train_model(name)
        return self.evaluate_all()

    def train_model(self, model_name: str) -> TrainingResult:
        """Train a single model by name ('fcn' | 'rnn' | 'lstm')."""
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
        evaluator = Evaluator(Path(cfg.output.results_dir))
        results = []

        for name in const.MODEL_NAMES:
            model, train_loader, val_loader, ckpt, _ = self._build_model_resources(
                name, bundle, cfg
            )
            tr = self._train_results.get(name)
            test_loader = (
                bundle.test_loader if name == const.MODEL_FCN else bundle.seq_test_loader
            )
            res = evaluator.evaluate(
                model, name, Path(ckpt),
                train_loader, val_loader, test_loader,
                epochs_trained=tr.epochs_trained if tr else 0,
                stopped_early=tr.stopped_early if tr else False,
            )
            self._eval_results[name] = res
            results.append(res)

        evaluator.save_comparison_table(results, Path(cfg.output.comparison_table_path))
        self._run_visualizations(bundle, results)
        return self._eval_results

    # ------------------------------------------------------------------ helpers

    def _load_data(self) -> DataBundle:
        """Load dataset.npz, fit z-score normaliser, return DataBundle."""
        cfg = self._cfg
        bundle = DataLoaderService(Path("data/dataset.npz"), cfg.training.batch_size).load()
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
            log = cfg.output.results_dir + "training_log_fcn.csv"
        elif name == const.MODEL_RNN:
            model = RNNModel(cfg.rnn)
            train_loader = bundle.seq_train_loader
            val_loader = bundle.seq_val_loader
            ckpt = cfg.output.rnn_checkpoint
            log = cfg.output.results_dir + "training_log_rnn.csv"
        else:
            model = LSTMModel(cfg.lstm)
            train_loader = bundle.seq_train_loader
            val_loader = bundle.seq_val_loader
            ckpt = cfg.output.lstm_checkpoint
            log = cfg.output.results_dir + "training_log_lstm.csv"
        return model, train_loader, val_loader, ckpt, log

    def _run_visualizations(self, bundle: DataBundle, results: list[EvalResult]) -> None:
        cfg = self._cfg
        viz = Visualizer(Path(cfg.output.results_dir))
        clean = bundle.y_train.flatten()
        noisy = bundle.X_train[:, 0].flatten()
        preds: dict[str, np.ndarray] = {}
        for name in const.MODEL_NAMES:
            model, _, _, ckpt, _ = self._build_model_resources(name, bundle, cfg)
            ckpt_path = Path(ckpt)
            if ckpt_path.exists():
                model.load_state_dict(
                    torch.load(ckpt_path, map_location="cpu", weights_only=True)
                )
            model.eval()
            loader = bundle.train_loader if name == const.MODEL_FCN else bundle.seq_train_loader
            preds[name] = self._predict_all(model, loader)

        viz.plot_clean_noisy_predicted(clean, noisy, preds)
        for name in const.MODEL_NAMES:
            tr = self._train_results.get(name)
            if tr:
                viz.plot_loss_curves(name, tr.train_losses, tr.val_losses)
        viz.plot_mse_comparison(results)

        pred_vs_actual: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in const.MODEL_NAMES:
            model, _, _, ckpt, _ = self._build_model_resources(name, bundle, cfg)
            ckpt_path = Path(ckpt)
            if ckpt_path.exists():
                model.load_state_dict(
                    torch.load(ckpt_path, map_location="cpu", weights_only=True)
                )
            model.eval()
            loader = bundle.test_loader if name == const.MODEL_FCN else bundle.seq_test_loader
            y_pred = self._predict_all(model, loader)
            y_true = bundle.y_test.flatten()
            viz.plot_residuals(name, y_true, y_pred)
            pred_vs_actual[name] = (y_true, y_pred)

        viz.plot_pred_vs_actual(pred_vs_actual)

    @staticmethod
    @torch.no_grad()
    def _predict_all(model: torch.nn.Module, loader: DataLoader) -> np.ndarray:  # type: ignore[type-arg]
        preds = []
        for xb, _ in loader:
            preds.append(model(xb).cpu().numpy())
        return np.concatenate(preds).flatten()
