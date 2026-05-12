"""Gap 12 integration tests — SensitivityAnalyzer creates expected output files."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.services.sensitivity import SensitivityAnalyzer


def _make_loaders():
    """Return synthetic (train_loader, val_loader) with FCN-shaped data."""
    x_tr = torch.randn(60, 15)
    y_tr = torch.randn(60, 1)
    x_va = torch.randn(20, 15)
    y_va = torch.randn(20, 1)
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=16)
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=16)
    return train_loader, val_loader


def _model_factory(cfg: dict) -> nn.Module:
    hs = cfg.get("hidden_size", 16)
    return nn.Sequential(nn.Linear(15, hs), nn.ReLU(), nn.Linear(hs, 1))


def _loader_factory(cfg: dict):
    return _make_loaders()


def _run_sweep(param_name: str, values: list, results_dir):
    base = {
        "learning_rate": 0.01, "sensitivity_epochs": 2,
        "hidden_size": 16, "dropout_rate": 0.0, "batch_size": 16,
        "window_size": 10, "hidden_size_rnn": 16,
    }
    analyzer = SensitivityAnalyzer(base, results_dir)
    results = analyzer.run(param_name, values, _model_factory, _loader_factory)
    analyzer.save_results(results)
    analyzer.plot_line_chart(param_name, results)
    analyzer.plot_heatmap(results)
    return results


def test_run_all_creates_sensitivity_window_size_png(tmp_path):
    """OAT sweep over window_size creates sensitivity_window_size.png."""
    _run_sweep("window_size", [5, 10, 20], tmp_path)
    assert (tmp_path / "sensitivity_window_size.png").exists()


def test_run_all_creates_sensitivity_hidden_size_png(tmp_path):
    """OAT sweep over hidden_size creates sensitivity_hidden_size.png."""
    _run_sweep("hidden_size", [16, 32, 64], tmp_path)
    assert (tmp_path / "sensitivity_hidden_size.png").exists()


def test_run_all_creates_sensitivity_learning_rate_png(tmp_path):
    """OAT sweep over learning_rate creates sensitivity_learning_rate.png."""
    _run_sweep("learning_rate", [0.0001, 0.001, 0.01], tmp_path)
    assert (tmp_path / "sensitivity_learning_rate.png").exists()


def test_run_all_creates_sensitivity_dropout_rate_png(tmp_path):
    """OAT sweep over dropout_rate creates sensitivity_dropout_rate.png."""
    _run_sweep("dropout_rate", [0.0, 0.1, 0.3], tmp_path)
    assert (tmp_path / "sensitivity_dropout_rate.png").exists()


def test_run_all_creates_sensitivity_batch_size_png(tmp_path):
    """OAT sweep over batch_size creates sensitivity_batch_size.png."""
    _run_sweep("batch_size", [16, 32, 64], tmp_path)
    assert (tmp_path / "sensitivity_batch_size.png").exists()


def test_run_all_creates_sensitivity_results_csv(tmp_path):
    """OAT sweep saves sensitivity_results.csv with expected columns."""
    import csv
    _run_sweep("learning_rate", [0.001, 0.01], tmp_path)
    path = tmp_path / "sensitivity_results.csv"
    assert path.exists()
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert all("param_name" in r and "param_value" in r and "val_mse" in r for r in rows)
    assert len(rows) == 2


def test_run_all_creates_sensitivity_heatmap_png(tmp_path):
    """OAT sweep creates sensitivity_heatmap.png."""
    _run_sweep("hidden_size", [16, 64], tmp_path)
    assert (tmp_path / "sensitivity_heatmap.png").exists()
