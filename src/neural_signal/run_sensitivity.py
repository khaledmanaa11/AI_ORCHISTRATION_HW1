"""Run OAT sensitivity analysis for all three model architectures.

Sweeps: learning_rate, hidden_size, batch_size, dropout_rate (FCN only).
Produces line charts + heatmap + CSV for each sweep in results/sensitivity/.

Usage:
    uv run python -m neural_signal.run_sensitivity
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_signal.models.fcn import FCNModel
from neural_signal.models.lstm import LSTMModel
from neural_signal.models.rnn import RNNModel
from neural_signal.services.sensitivity import SensitivityAnalyzer, SensitivityResult
from neural_signal.shared.config_types import FCNConfig, LSTMConfig, RNNConfig

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "dataset.npz"
SENS_DIR = ROOT / "results" / "sensitivity"
SENS_EPOCHS = 20  # quick but enough to show trends


# ── data helpers ─────────────────────────────────────────────────────────────

def _load_data() -> dict[str, np.ndarray]:
    data = np.load(DATA_PATH)
    return {k: data[k].astype(np.float32) for k in data.files}


def _make_fcn_loaders(
    data: dict, batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    x_tr = np.concatenate([data["X_train"], data["C_train"]], axis=1)
    x_va = np.concatenate([data["X_val"], data["C_val"]], axis=1)
    y_tr = data["y_train"].reshape(-1, 1)
    y_va = data["y_val"].reshape(-1, 1)
    tr = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
                    batch_size=batch_size, shuffle=False)
    return tr, va


def _make_seq_loaders(
    data: dict, batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    def _build(x_win, c):
        n, w = x_win.shape
        x3 = x_win.reshape(n, w, 1)
        c3 = np.broadcast_to(c[:, np.newaxis, :], (n, w, c.shape[1])).copy()
        return np.concatenate([x3, c3], axis=2)

    x_tr = _build(data["X_train"], data["C_train"])
    x_va = _build(data["X_val"], data["C_val"])
    y_tr = data["y_train"].reshape(-1, 1)
    y_va = data["y_val"].reshape(-1, 1)
    tr = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
                    batch_size=batch_size, shuffle=False)
    return tr, va


# ── model factories ──────────────────────────────────────────────────────────

def _fcn_factory(cfg: dict) -> nn.Module:
    hs = cfg.get("hidden_size", 64)
    return FCNModel(FCNConfig(
        hidden_sizes=[hs * 2, hs],
        dropout_rate=cfg.get("dropout_rate", 0.1),
        output_size=1,
    ))


def _rnn_factory(cfg: dict) -> nn.Module:
    return RNNModel(RNNConfig(
        hidden_size=cfg.get("hidden_size", 64),
        nonlinearity="tanh",
        num_layers=1,
        output_size=1,
        input_size=6,
    ))


def _lstm_factory(cfg: dict) -> nn.Module:
    hs = cfg.get("hidden_size", 64)
    return LSTMModel(LSTMConfig(
        hidden_size=hs,
        num_layers=1,
        dense_hidden_size=max(hs // 2, 8),
        output_size=1,
        input_size=6,
    ))


# ── plotting ─────────────────────────────────────────────────────────────────

def _plot_multi_model(
    param_name: str,
    values: list,
    model_results: dict[str, list[SensitivityResult]],
    out_dir: Path,
) -> Path:
    """Line chart overlaying all three models for one swept parameter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"FCN": "#e74c3c", "RNN": "#e67e22", "LSTM": "#9b59b6"}
    for label, results in model_results.items():
        mses = [r.val_mse for r in results]
        ax.plot(values, mses, "o-", label=label, color=colors.get(label, "blue"),
                linewidth=2, markersize=6)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Val MSE", fontsize=12)
    ax.set_title(f"Sensitivity: {param_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if param_name == "learning_rate":
        ax.set_xscale("log")
    path = out_dir / f"sensitivity_{param_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_summary_heatmap(
    all_results: dict[str, dict[str, list[SensitivityResult]]],
    out_dir: Path,
) -> Path:
    """Heatmap: rows = parameters, columns = models, cell = best val MSE."""
    params = list(all_results.keys())
    models = ["FCN", "RNN", "LSTM"]
    matrix = np.zeros((len(params), len(models)))
    for i, param in enumerate(params):
        for j, model in enumerate(models):
            results = all_results[param].get(model, [])
            if results:
                matrix[i, j] = min(r.val_mse for r in results)

    fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.8)))
    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, fontsize=11)
    for i in range(len(params)):
        for j in range(len(models)):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.mean() else "black")
    fig.colorbar(im, ax=ax, label="Best Val MSE", shrink=0.8)
    ax.set_title("OAT Sensitivity — Best MSE per Sweep", fontsize=14, fontweight="bold")
    path = out_dir / "sensitivity_summary_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    torch.manual_seed(42)
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading dataset …")
    data = _load_data()

    base_cfg = {
        "learning_rate": 0.001,
        "hidden_size": 64,
        "batch_size": 64,
        "dropout_rate": 0.1,
        "sensitivity_epochs": SENS_EPOCHS,
    }

    sweeps = {
        "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01],
        "hidden_size": [16, 32, 64, 128],
        "batch_size": [16, 32, 64, 128, 256],
    }

    all_results: dict[str, dict[str, list[SensitivityResult]]] = {}

    for param_name, values in sweeps.items():
        print(f"\n{'='*60}")
        print(f"Sweeping: {param_name} = {values}")
        print(f"{'='*60}")
        model_results: dict[str, list[SensitivityResult]] = {}

        for model_label, model_fn, loader_fn in [
            ("FCN", _fcn_factory, lambda cfg: _make_fcn_loaders(data, cfg.get("batch_size", 64))),
            ("RNN", _rnn_factory, lambda cfg: _make_seq_loaders(data, cfg.get("batch_size", 64))),
            ("LSTM", _lstm_factory, lambda cfg: _make_seq_loaders(data, cfg.get("batch_size", 64))),
        ]:
            print(f"  {model_label}: ", end="", flush=True)
            analyzer = SensitivityAnalyzer(base_cfg, SENS_DIR)
            results = analyzer.run(param_name, values, model_fn, loader_fn)
            model_results[model_label] = results
            analyzer.save_results(results)
            best = min(results, key=lambda r: r.val_mse)
            print(f"best {param_name}={best.param_value} -> MSE={best.val_mse:.4f}")

        _plot_multi_model(param_name, values, model_results, SENS_DIR)
        all_results[param_name] = model_results

    # FCN-only sweep: dropout_rate
    print(f"\n{'='*60}")
    print("Sweeping: dropout_rate (FCN only)")
    print(f"{'='*60}")
    dropout_vals = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    analyzer = SensitivityAnalyzer(base_cfg, SENS_DIR)
    dropout_results = analyzer.run(
        "dropout_rate", dropout_vals, _fcn_factory,
        lambda cfg: _make_fcn_loaders(data, cfg.get("batch_size", 64)),
    )
    analyzer.save_results(dropout_results)
    best_do = min(dropout_results, key=lambda r: r.val_mse)
    print(f"  FCN: best dropout={best_do.param_value} -> MSE={best_do.val_mse:.4f}")

    # Plot dropout as single-model chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dropout_vals, [r.val_mse for r in dropout_results], "o-",
            color="#e74c3c", linewidth=2, markersize=6, label="FCN")
    ax.set_xlabel("dropout_rate", fontsize=12)
    ax.set_ylabel("Val MSE", fontsize=12)
    ax.set_title("Sensitivity: dropout_rate (FCN only)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.savefig(SENS_DIR / "sensitivity_dropout_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    all_results["dropout_rate"] = {"FCN": dropout_results}

    # Summary heatmap
    _plot_summary_heatmap(all_results, SENS_DIR)

    print(f"\nDone! All sensitivity results saved to {SENS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
