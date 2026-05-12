# PLAN — Architecture & Design (FCN · RNN · LSTM)

**Version:** 1.00
**Date:** 2026-05-10
**Status:** Awaiting Approval

---

## 1. Technology Stack

| Concern | Choice | Reason |
|---------|--------|--------|
| Language | Python 3.10+ | Required by guidelines |
| Package manager | uv | Mandatory per CLAUDE.md |
| Linter | Ruff | Mandatory per CLAUDE.md |
| Deep learning | PyTorch | Model definitions, training loop, dataloaders |
| Numerics | NumPy | Data loading and preprocessing |
| Data | Pandas | Results tables, CSV export |
| Visualisation | Matplotlib + Seaborn | Static plots, loss curves |
| Notebooks | Jupyter | Required by guidelines |
| Testing | pytest + pytest-cov + pytest-mock | TDD mandatory |

---

## 2. Repository Structure

```
HW1/
├── src/neural_signal/
│   ├── __init__.py                    # __all__, __version__
│   ├── sdk/
│   │   └── sdk.py                     # NeuralSignalSDK — single public entry point
│   ├── models/
│   │   ├── fcn.py                     # FCNModel (Dense 128→64→1, Dropout, L2)
│   │   ├── rnn.py                     # RNNModel (hidden=64, tanh, many-to-one)
│   │   └── lstm.py                    # LSTMModel (hidden=64, Dense 32→1, many-to-one)
│   ├── services/
│   │   ├── data_loader.py             # Load dataset.npz, build DataLoaders
│   │   ├── preprocessor.py            # Z-score normalization, tensor conversion
│   │   ├── trainer.py                 # Training loop, Adam, MSE, early stopping
│   │   ├── evaluator.py               # MSE eval, comparison table
│   │   └── visualizer.py             # All plots (clean/noisy/pred, loss curves, etc.)
│   ├── shared/
│   │   ├── gatekeeper.py              # ApiGatekeeper (mandatory)
│   │   ├── config.py                  # ConfigManager — loads setup.json
│   │   └── version.py                 # __version__ = "1.00"
│   ├── constants.py                   # Model names, keys, hyperparameter names
│   └── main.py                        # CLI entry point (calls SDK only)
├── tests/
│   ├── unit/
│   │   ├── test_fcn.py
│   │   ├── test_rnn.py
│   │   ├── test_lstm.py
│   │   ├── test_data_loader.py
│   │   ├── test_preprocessor.py
│   │   ├── test_trainer.py
│   │   ├── test_evaluator.py
│   │   ├── test_visualizer.py
│   │   ├── test_config.py
│   │   ├── test_gatekeeper.py
│   │   └── test_sdk.py
│   ├── integration/
│   │   └── test_pipeline.py           # end-to-end: load → train → eval → plot
│   └── conftest.py                    # shared fixtures
├── config/
│   ├── setup.json                     # All hyperparameters and paths
│   └── rate_limits.json               # Gatekeeper rate limits
├── data/                              # dataset.npz (from Phase 1)
├── results/                           # charts, CSVs, checkpoints
├── assets/
├── notebooks/
│   └── results_analysis.ipynb
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── TODO.md
│   ├── PRD_fcn.md
│   ├── PRD_rnn.md
│   └── PRD_lstm.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env-example
└── .gitignore
```

---

## 3. Module Responsibilities

### 3.1 `sdk/sdk.py` — `NeuralSignalSDK`
Single public interface. No business logic here — orchestration only.

```
NeuralSignalSDK
  ├── run_all()          → trains + evaluates all 3 models, saves all results
  ├── train_model(name)  → trains a single model by name ("fcn"|"rnn"|"lstm")
  └── evaluate_all()     → loads best checkpoints, runs eval, saves comparison table
```

### 3.2 `services/data_loader.py` — `DataLoaderService`
- **Input**: path to `data/dataset.npz`, batch_size, window_size from config
- **Output**: `DataBundle` (train/val/test `DataLoader` objects + raw arrays for visualization)
- **Logic**:
  - Load npz, reconstruct `X`, `C`, `y` splits
  - Concatenate `X` and `C` → `x_input` shape `(N, 15)` for FCN
  - Reshape `X` → `(N, seq_len, 1)` for RNN/LSTM
  - Wrap in `TensorDataset` → `DataLoader(shuffle=True for train, False for val/test)`

### 3.3 `services/preprocessor.py` — `Preprocessor`
- **Input**: raw numpy arrays `X_train`, `X_val`, `X_test`
- **Output**: normalized arrays, fitted scaler parameters (mean, std)
- **Logic**: z-score normalization fitted on train set only, applied to val/test
- **Saves**: scaler params to `results/scaler_params.json` for reproducibility

### 3.4 `models/fcn.py` — `FCNModel(nn.Module)`
- Input: `(batch, 15)`
- Layers: `Linear(15,128)` → `ReLU` → `Dropout(p)` → `Linear(128,64)` → `ReLU` → `Dropout(p)` → `Linear(64,1)`
- L2 via `weight_decay=1e-4` in Adam optimizer (not as a separate layer)
- All hyperparams from config — no literals in model code

### 3.5 `models/rnn.py` — `RNNModel(nn.Module)`
- Input: `(batch, seq_len, input_size)`
- Layers: `nn.RNN(input_size, 64, nonlinearity='tanh', batch_first=True)` → `nn.Linear(64, 1)`
- Forward: run RNN, extract `output[:, -1, :]` (last time step), apply linear head
- Many-to-one architecture

### 3.6 `models/lstm.py` — `LSTMModel(nn.Module)`
- Input: `(batch, seq_len, input_size)`
- Layers: `nn.LSTM(input_size, 64, batch_first=True)` → `nn.Linear(64, 32)` → `ReLU` → `nn.Linear(32, 1)`
- Forward: run LSTM, extract `output[:, -1, :]`, apply dense head
- Many-to-one architecture

### 3.7 `services/trainer.py` — `Trainer`
- **Input**: model, train DataLoader, val DataLoader, training config
- **Output**: `TrainingResult` (best_val_mse, epochs_trained, stopped_early, checkpoint_path)
- **Logic**:
  - Epoch loop: forward → MSE loss → backward → Adam step
  - Val loop every epoch (no grad): compute val MSE
  - Early stopping: track best val loss, increment patience counter on no improvement
  - Save best checkpoint: `torch.save(model.state_dict(), checkpoint_path)`
  - Log per-epoch metrics to CSV

### 3.8 `services/evaluator.py` — `Evaluator`
- **Input**: model (with loaded checkpoint), test DataLoader
- **Output**: `EvalResult` (train_mse, val_mse, test_mse)
- **Logic**: run inference on all splits, compute MSE, build comparison DataFrame

### 3.9 `services/visualizer.py` — `Visualizer`
- **Input**: clean signals, noisy signals, predictions from all three models, training logs
- **Output**: PNG files in `results/`
- **Logic**: matplotlib figure construction per plot spec in FR-007

### 3.10 `shared/config.py` — `ConfigManager`
- Loads and validates `config/setup.json` and `config/rate_limits.json`
- Returns typed dataclasses covering: dataset paths, model hyperparams, training params, output paths
- Raises `ConfigValidationError` on invalid values

### 3.11 `shared/gatekeeper.py` — `ApiGatekeeper`
- FIFO queue for all external/rate-limited calls
- Rate-limit enforcement from `rate_limits.json`
- Used for any file I/O or future external model API calls

---

## 4. Data Flow

```
data/dataset.npz
      │
      ▼
DataLoaderService
      │  ├─ X_train (N_train, W=10), C_train (N_train, 5), y_train (N_train,1)
      │  └─ x_input = concat([X, C]) → (N, 15) for FCN
      │     x_seq   = X.reshape(N, 10, 1)       for RNN/LSTM
      ▼
Preprocessor  ── z-score on X features (train stats only)
      │
      ▼
┌─────────────────────────────────────────────┐
│              Training Loop (Trainer)         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  FCNModel │  │  RNNModel │  │ LSTMModel│  │
│  │  Adam+L2  │  │   Adam    │  │   Adam   │  │
│  │  MSE loss │  │  MSE loss │  │ MSE loss │  │
│  │  EarlyStop│  │  EarlyStop│  │ EarlyStop│  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘
      │
      ▼
Evaluator  ─────► results/comparison_table.csv
      │
      ▼
Visualizer ─────► results/clean_noisy_predicted.png
                  results/loss_curves_<model>.png
                  results/mse_comparison.png
                  results/residuals_<model>.png
                  results/pred_vs_actual.png
```

---

## 5. Input Encoding per Model

```
For window k with window size W=10 and C = [0,1,0,0,0]:

FCN INPUT (flat):
  x_input = [s5_noisy[k], …, s5_noisy[k+9],  0, 1, 0, 0, 0]
  shape    = (15,)

RNN / LSTM INPUT (sequential):
  x_seq = s5_noisy[k:k+10].reshape(10, 1)    ← sequence of 10 scalar samples
  C is prepended as the first time-step features OR concatenated at each step
  shape = (10, 1) or (10, 6) depending on C injection strategy
  (configuration choice — document in PRD_rnn.md / PRD_lstm.md)

LABEL:
  y = s2_clean[k + 5]     ← clean s2 value at window center
  shape = (1,)
```

---

## 6. Model Hyperparameter Config (`config/setup.json` additions)

```json
{
  "training": {
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "max_epochs": 200,
    "early_stopping_patience": 10,
    "val_split": 0.20,
    "random_seed": 42
  },
  "fcn": {
    "hidden_sizes": [128, 64],
    "dropout_rate": 0.1,
    "output_size": 1
  },
  "rnn": {
    "hidden_size": 64,
    "nonlinearity": "tanh",
    "num_layers": 1,
    "output_size": 1
  },
  "lstm": {
    "hidden_size": 64,
    "num_layers": 1,
    "dense_hidden_size": 32,
    "output_size": 1
  },
  "output": {
    "results_dir": "results/",
    "checkpoint_dir": "results/checkpoints/",
    "fcn_checkpoint": "results/checkpoints/fcn_best.pt",
    "rnn_checkpoint": "results/checkpoints/rnn_best.pt",
    "lstm_checkpoint": "results/checkpoints/lstm_best.pt"
  }
}
```

---

## 7. File Size Budget (≤ 150 lines each)

| File | Est. lines |
|------|-----------|
| sdk.py | ~90 |
| models/fcn.py | ~70 |
| models/rnn.py | ~70 |
| models/lstm.py | ~80 |
| services/data_loader.py | ~100 |
| services/preprocessor.py | ~80 |
| services/trainer.py | ~140 |
| services/evaluator.py | ~90 |
| services/visualizer.py | ~140 |
| shared/config.py | ~130 |
| shared/gatekeeper.py | ~120 |
| shared/version.py | ~20 |
| constants.py | ~40 |
| main.py | ~60 |

All under 150. `trainer.py` and `visualizer.py` are close — split into mixins if exceeded.

---

## 8. Testing Strategy (TDD — RED → GREEN → REFACTOR)

| Test file | Covers |
|-----------|--------|
| `test_fcn.py` | Forward pass shape, dropout, weight decay, no-NaN output |
| `test_rnn.py` | Forward pass shape, last-hidden-state extraction, tanh activations |
| `test_lstm.py` | Forward pass shape, last-output extraction, dense head |
| `test_data_loader.py` | DataLoader batch shape, concatenation, reshaping, shuffle behavior |
| `test_preprocessor.py` | Z-score stats, no data leakage from val/test into train fit |
| `test_trainer.py` | Loss decreases over epochs, early stopping fires at correct patience |
| `test_evaluator.py` | MSE computation, comparison table shape and columns |
| `test_visualizer.py` | Files created, figure dimensions, no errors on mock data |
| `test_config.py` | Load, validate, reject bad values, all new training/model fields |
| `test_gatekeeper.py` | Execute, retry, queue depth, logging |
| `test_sdk.py` | Orchestration calls all services, returns correct result types |
| `test_pipeline.py` (integration) | Full run: load → train → eval → plot → files exist |

Target: ≥ 85% coverage enforced by `fail_under = 85`.

---

## 9. Visualisations (saved to `results/`)

1. `results/clean_noisy_predicted.png` — **mandatory** — clean + noisy + all 3 predictions overlaid
2. `results/loss_curves_fcn.png` — train vs. val MSE per epoch for FCN
3. `results/loss_curves_rnn.png` — train vs. val MSE per epoch for RNN
4. `results/loss_curves_lstm.png` — train vs. val MSE per epoch for LSTM
5. `results/mse_comparison.png` — grouped bar chart: train/val/test MSE per model
6. `results/residuals_fcn.png` — residuals (y_pred - y_true) for FCN on test set
7. `results/residuals_rnn.png` — residuals for RNN
8. `results/residuals_lstm.png` — residuals for LSTM
9. `results/pred_vs_actual.png` — scatter: predicted vs. actual (all 3 models, 3 panels)
10. `results/sensitivity/` — OAT parameter sweep plots (heatmaps, line charts)
