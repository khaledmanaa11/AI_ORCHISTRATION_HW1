"""Unit tests for services/sensitivity.py — SensitivityAnalyzer (OAT)."""

from __future__ import annotations

import torch.nn as nn

from neural_signal.services.sensitivity import SensitivityAnalyzer, SensitivityResult

_BASE_CFG = {
    "learning_rate": 0.01,
    "sensitivity_epochs": 2,
}


def _model_factory(cfg: dict) -> nn.Module:
    return nn.Linear(15, 1)


def _loader_factory(cfg: dict):
    return SensitivityAnalyzer.make_simple_loaders(n_train=60, n_val=20, batch_size=32)


def test_sensitivity_analyzer_instantiates(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    assert sa is not None


def test_run_returns_list(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.01, 0.001], _model_factory, _loader_factory)
    assert isinstance(results, list)


def test_run_result_count_matches_values(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    values = [0.1, 0.01, 0.001]
    results = sa.run("learning_rate", values, _model_factory, _loader_factory)
    assert len(results) == len(values)


def test_result_is_sensitivity_result(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.01], _model_factory, _loader_factory)
    assert isinstance(results[0], SensitivityResult)


def test_result_param_name_matches(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.01], _model_factory, _loader_factory)
    assert results[0].param_name == "learning_rate"


def test_result_param_value_matches(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.005], _model_factory, _loader_factory)
    assert results[0].param_value == 0.005


def test_val_mse_is_finite(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.01], _model_factory, _loader_factory)
    assert results[0].val_mse < float("inf")


def test_val_mse_is_non_negative(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.01], _model_factory, _loader_factory)
    assert results[0].val_mse >= 0.0


def test_base_config_not_mutated(tmp_path):
    """Original base_config dict must not be modified by run()."""
    base = {"learning_rate": 0.01, "sensitivity_epochs": 2}
    sa = SensitivityAnalyzer(base, tmp_path)
    sa.run("learning_rate", [0.1, 0.001], _model_factory, _loader_factory)
    assert base["learning_rate"] == 0.01


def test_results_dir_created_if_missing(tmp_path):
    new_dir = tmp_path / "oat_results"
    SensitivityAnalyzer(_BASE_CFG, new_dir)
    assert new_dir.exists()


def test_make_simple_loaders_returns_two_loaders(tmp_path):
    train_l, val_l = SensitivityAnalyzer.make_simple_loaders()
    assert train_l is not None and val_l is not None


def test_make_simple_loaders_respects_batch_size(tmp_path):
    train_l, _ = SensitivityAnalyzer.make_simple_loaders(n_train=64, batch_size=16)
    xb, _ = next(iter(train_l))
    assert xb.shape[0] <= 16


def test_save_results_creates_csv(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.01], _model_factory, _loader_factory)
    path = sa.save_results(results)
    assert path.exists() and path.suffix == ".csv"


def test_csv_has_correct_columns(tmp_path):
    import csv as _csv
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.01], _model_factory, _loader_factory)
    path = sa.save_results(results)
    with path.open() as fh:
        header = next(_csv.reader(fh))
    assert "param_name" in header and "param_value" in header and "val_mse" in header


def test_plot_line_chart_creates_file(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.01, 0.001], _model_factory, _loader_factory)
    path = sa.plot_line_chart("learning_rate", results)
    assert path.exists() and path.stat().st_size > 0


def test_plot_heatmap_creates_file(tmp_path):
    sa = SensitivityAnalyzer(_BASE_CFG, tmp_path)
    results = sa.run("learning_rate", [0.01, 0.001], _model_factory, _loader_factory)
    path = sa.plot_heatmap(results)
    assert path.exists() and path.stat().st_size > 0
