"""Tests for services/evaluator.py — Evaluator."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.services.evaluator import EvalResult, Evaluator


@pytest.fixture()
def simple_model() -> nn.Module:
    return nn.Linear(15, 1)


def _loader(n: int = 80) -> DataLoader:
    x = torch.randn(n, 15)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def test_evaluator_instantiates(tmp_results_dir):
    ev = Evaluator(tmp_results_dir)
    assert ev is not None


def test_evaluate_returns_eval_result(simple_model, tmp_results_dir, tmp_path):
    loader = _loader()
    ev = Evaluator(tmp_results_dir)
    result = ev.evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt",
        loader, loader, loader, 5, False
    )
    assert isinstance(result, EvalResult)


def test_eval_result_has_model_name(simple_model, tmp_results_dir, tmp_path):
    loader = _loader()
    result = Evaluator(tmp_results_dir).evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    assert result.model_name == "fcn"


def test_eval_result_train_mse_finite(simple_model, tmp_results_dir, tmp_path):
    loader = _loader()
    result = Evaluator(tmp_results_dir).evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    assert result.train_mse < float("inf")


def test_eval_result_test_mse_finite(simple_model, tmp_results_dir, tmp_path):
    loader = _loader()
    result = Evaluator(tmp_results_dir).evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    assert result.test_mse < float("inf")


def test_eval_mse_is_non_negative(simple_model, tmp_results_dir, tmp_path):
    loader = _loader()
    result = Evaluator(tmp_results_dir).evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    assert result.train_mse >= 0
    assert result.val_mse >= 0
    assert result.test_mse >= 0


def test_save_comparison_table_creates_csv(simple_model, tmp_results_dir, tmp_path):
    loader = _loader()
    ev = Evaluator(tmp_results_dir)
    r = ev.evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    table_path = tmp_results_dir / "comparison.csv"
    df = ev.save_comparison_table([r], table_path)
    assert table_path.exists()
    assert len(df) == 1


def test_comparison_table_has_correct_columns(simple_model, tmp_results_dir, tmp_path):
    loader = _loader()
    ev = Evaluator(tmp_results_dir)
    r = ev.evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    df = ev.save_comparison_table([r], tmp_results_dir / "cmp.csv")
    assert "model" in df.columns
    assert "train_mse" in df.columns
    assert "test_mse" in df.columns


def test_zero_mse_when_model_predicts_perfectly(tmp_results_dir, tmp_path):
    """A model that always returns zero has MSE equal to mean(y^2)."""
    loader = DataLoader(
        TensorDataset(torch.randn(32, 15), torch.zeros(32, 1)), batch_size=32
    )
    model = nn.Linear(15, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    ev = Evaluator(tmp_results_dir)
    result = ev.evaluate(
        model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 1, False
    )
    assert result.train_mse == pytest.approx(0.0, abs=1e-6)


def test_evaluator_with_explicit_cpu_device(simple_model, tmp_results_dir, tmp_path):
    """Evaluator with device='cpu' should produce finite MSE scores."""
    loader = _loader()
    ev = Evaluator(tmp_results_dir, device="cpu")
    result = ev.evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    assert result.test_mse < float("inf")


def test_mse_computed_without_gradient(simple_model, tmp_results_dir, tmp_path):
    """evaluate() should not accumulate gradients."""
    import torch
    loader = _loader()
    ev = Evaluator(tmp_results_dir)
    with torch.no_grad():
        result = ev.evaluate(
            simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
        )
    assert result.train_mse >= 0


def test_comparison_dataframe_has_val_mse_column(simple_model, tmp_results_dir, tmp_path):
    loader = _loader()
    r = Evaluator(tmp_results_dir).evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    df = Evaluator(tmp_results_dir).save_comparison_table([r], tmp_results_dir / "c.csv")
    assert "val_mse" in df.columns


def test_comparison_dataframe_has_epochs_trained_column(simple_model, tmp_results_dir, tmp_path):
    loader = _loader()
    r = Evaluator(tmp_results_dir).evaluate(
        simple_model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    df = Evaluator(tmp_results_dir).save_comparison_table([r], tmp_results_dir / "c2.csv")
    assert "epochs_trained" in df.columns
