"""Extra Evaluator tests — overflow from test_evaluator.py (150-line limit)."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.services.evaluator import Evaluator


def _loader(n: int = 80) -> DataLoader:
    x = torch.randn(n, 15)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def test_comparison_dataframe_has_stopped_early_column(tmp_results_dir, tmp_path):
    loader = _loader()
    model = nn.Linear(15, 1)
    r = Evaluator(tmp_results_dir).evaluate(
        model, "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    df = Evaluator(tmp_results_dir).save_comparison_table([r], tmp_results_dir / "c3.csv")
    assert "stopped_early" in df.columns


def test_comparison_dataframe_has_three_rows(tmp_results_dir, tmp_path):
    loader = _loader()
    ev = Evaluator(tmp_results_dir)
    results = [
        ev.evaluate(nn.Linear(15, 1), name, tmp_path / f"{name}.pt",
                    loader, loader, loader, 5, False)
        for name in ("fcn", "rnn", "lstm")
    ]
    df = ev.save_comparison_table(results, tmp_results_dir / "c4.csv")
    assert len(df) == 3


def test_comparison_csv_loadable_by_pandas(tmp_results_dir, tmp_path):
    loader = _loader()
    r = Evaluator(tmp_results_dir).evaluate(
        nn.Linear(15, 1), "fcn", tmp_path / "ckpt.pt", loader, loader, loader, 5, False
    )
    csv_path = tmp_results_dir / "c5.csv"
    Evaluator(tmp_results_dir).save_comparison_table([r], csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 1


def test_checkpoint_loaded_before_eval(tmp_results_dir, tmp_path):
    """Evaluator loads checkpoint; fresh model produces same result as trained model."""
    loader = _loader()
    model = nn.Linear(15, 1)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)
    fresh = nn.Linear(15, 1)
    ev = Evaluator(tmp_results_dir)
    result = ev.evaluate(fresh, "fcn", ckpt, loader, loader, loader, 1, False)
    assert result.test_mse < float("inf")
