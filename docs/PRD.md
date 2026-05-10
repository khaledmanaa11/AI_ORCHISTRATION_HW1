# PRD — Neural Network Signal Regression (FCN · RNN · LSTM)

**Version:** 1.00
**Date:** 2026-05-10
**Author:** Khaled Mnaa
**Course:** AI Orchestration — Dr. Segal Yoram
**Status:** Awaiting Approval

---

## 1. Purpose

Train and compare three neural network architectures on the noisy sine-signal dataset generated
in HW1 Phase 1. Each model learns to predict the **clean signal value** from a noisy composite
window plus a one-hot signal selector. The three architectures are:

| Model | Acronym | Key Characteristic |
|-------|---------|-------------------|
| Fully Connected Network | FCN | Dense layers, no temporal memory |
| Vanilla Recurrent Neural Network | RNN | Shared weights across time steps, tanh |
| Long Short-Term Memory | LSTM | Gated memory, richer temporal modeling |

---

## 2. Background & Motivation

The dataset (from Phase 1) consists of:
- `X` — sliding windows of the composite noisy signal `s5_noisy`, shape `(N, W)` where `W=10`
- `C` — one-hot signal selectors of length 5, shape `(N, 5)`
- `y` — scalar clean value of the C-selected signal at window center, shape `(N, 1)`

At model forward-pass time the model input is `concat([X_window, C])` → shape `(15,)` for FCN,
or the window is reshaped to `(seq_len, features)` for RNN/LSTM.

The goal is to benchmark which architecture best recovers the clean signal under this
conditioning setup, using MSE as the evaluation metric.

---

## 3. Scope

### In Scope
- Loading the pre-generated `data/dataset.npz` (X, C, y splits)
- Preprocessing: normalization, tensor conversion, DataLoader construction
- FCN model: Dense(128,ReLU) → Dense(64,ReLU) → Dense(1,Linear), Dropout(0.1), L2 weight decay 1e-4
- Vanilla RNN model: RNN(hidden=64, tanh) → FC(1), many-to-one (last hidden state)
- LSTM model: LSTM(hidden=64) → Dense(32,ReLU) → Dense(1,Linear), many-to-one
- Training pipeline: Adam (lr=0.001), MSE loss, batch size 64, validation split 20%, early stopping
- Evaluation: train MSE, val MSE, final test MSE, per-epoch loss curves
- Unified visualization: "Clean vs. Noisy vs. Predicted" — all three models on one graph
- Full SDK architecture, ApiGatekeeper, TDD, Ruff compliance, uv package manager
- OAT parameter sensitivity analysis (window_size, hidden_size, lr, dropout, batch_size)
- Jupyter notebook with results and statistical comparison
- Per-algorithm PRDs (docs/PRD_fcn.md, docs/PRD_rnn.md, docs/PRD_lstm.md)
- README, API docs, prompt engineering log

### Out of Scope
- Transformer / attention architectures
- Hyperparameter search / AutoML
- Real-time streaming inference
- GUI frontend
- Re-generating the dataset (Phase 1 already complete)

---

## 4. Functional Requirements

### FR-001: Data Loading
- System shall load `data/dataset.npz` using the DataLoaderService
- System shall concatenate X and C into model input: `x_input = concat([X_window, C])` → `(15,)`
- System shall wrap train/val/test arrays in PyTorch `TensorDataset` and `DataLoader`
- Batch size shall be read from `config/setup.json` (default 64)
- System shall apply z-score normalization to X features using train-set statistics only

### FR-002: FCN Model (`models/fcn.py`)
- Architecture: `Input(15)` → `Dense(128, ReLU)` → `Dropout(0.1)` → `Dense(64, ReLU)` → `Dropout(0.1)` → `Dense(1, Linear)`
- L2 weight decay `1e-4` applied via optimizer `weight_decay` parameter
- Dropout rate configurable from `config/setup.json`
- Input shape: `(batch, 15)` — flat concatenated window + selector
- Output shape: `(batch, 1)` — scalar prediction

### FR-003: Vanilla RNN Model (`models/rnn.py`)
- Architecture: `RNN(input_size=1, hidden_size=64, nonlinearity='tanh', batch_first=True)` → `Linear(64, 1)`
- Many-to-one: extract only last hidden state `h_t[:, -1, :]`
- Input shape: `(batch, seq_len=10, 1)` — window reshaped; C concatenated as extra feature channels or pre-pended
- Output shape: `(batch, 1)`

### FR-004: LSTM Model (`models/lstm.py`)
- Architecture: `LSTM(input_size=1, hidden_size=64, batch_first=True)` → `Linear(64, 32)` → `ReLU` → `Linear(32, 1)`
- Many-to-one: extract only last output `out[:, -1, :]`
- Input shape: `(batch, seq_len=10, 1)`
- Output shape: `(batch, 1)`

### FR-005: Training Pipeline (`services/trainer.py`)
- Optimizer: Adam, initial lr=0.001, weight_decay=1e-4 for FCN
- Loss: MSE (`nn.MSELoss`)
- Batch size: 64 (from config)
- Validation split: 20% of training data held out before training (from pre-split in dataset.npz)
- Early stopping: monitor validation loss, patience configurable (default=10), stop if no improvement
- Maximum epochs: configurable (default=200)
- Checkpoint: save best model weights by minimum validation MSE
- Log per-epoch train/val loss to `results/training_log_<model>.csv`

### FR-006: Evaluation (`services/evaluator.py`)
- Compute final train, validation, and test MSE for each model
- Produce comparison table: `model | train_mse | val_mse | test_mse | epochs_trained | stopped_early`
- Save table to `results/comparison_table.csv`

### FR-007: Visualization (`services/visualizer.py`)
- **Plot 1 (mandatory)**: "Clean vs. Noisy vs. Predicted" — single figure, overlay:
  - Clean signal (green, solid)
  - Noisy signal (grey, dashed)
  - FCN prediction (red)
  - RNN prediction (orange)
  - LSTM prediction (purple)
  - Save to `results/clean_noisy_predicted.png`
- **Plot 2**: Training loss curves per model (train vs. val per epoch), saved to `results/loss_curves_<model>.png`
- **Plot 3**: MSE comparison bar chart (all three models, train/val/test), saved to `results/mse_comparison.png`
- **Plot 4**: Residuals plot per model, saved to `results/residuals_<model>.png`
- **Plot 5**: Scatter plot of predicted vs. actual for all three models, saved to `results/pred_vs_actual.png`

### FR-008: SDK & Architecture
- All logic exposed through `sdk/sdk.py` as `NeuralSignalSDK`
- `NeuralSignalSDK.run_all()` trains and evaluates all three models sequentially
- `NeuralSignalSDK.train_model(model_name)` trains a single model by name
- `NeuralSignalSDK.evaluate_all()` loads checkpoints and re-evaluates
- No business logic in `main.py` — CLI calls SDK only
- All config loaded from `config/setup.json` and `config/rate_limits.json`
- ApiGatekeeper wraps any external or file I/O that is rate-limited

### FR-009: Parameter Sensitivity (OAT)
- Vary one parameter at a time: window_size, hidden_size, learning_rate, dropout_rate, batch_size
- For each parameter value: train all three models, record val MSE
- Save sensitivity results to `results/sensitivity/`
- Plot heatmaps and line charts of MSE vs. parameter value

---

## 5. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-001 | Test coverage | ≥ 85% (pytest-cov, fail_under=85) |
| NFR-002 | Linting | 0 Ruff errors |
| NFR-003 | File length | ≤ 150 lines per file |
| NFR-004 | Package manager | uv only |
| NFR-005 | No hardcoded values | All from config JSON |
| NFR-006 | Secrets | Never in source; `.env-example` committed |
| NFR-007 | Reproducibility | Fixed random seed for weight init and data split |
| NFR-008 | Framework | PyTorch (torch, torch.nn, torch.optim) |

---

## 6. Architecture Constraints

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`)
- **Data**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Notebook**: Jupyter
- **Config**: JSON files in `config/`
- **Testing**: pytest, pytest-cov, pytest-mock
- **Linting**: Ruff
- **Package manager**: uv

---

## 7. Deliverables

| Deliverable | Location |
|-------------|----------|
| This PRD | `docs/PRD.md` |
| Architecture Plan | `docs/PLAN.md` |
| Task List | `docs/TODO.md` |
| FCN PRD | `docs/PRD_fcn.md` |
| RNN PRD | `docs/PRD_rnn.md` |
| LSTM PRD | `docs/PRD_lstm.md` |
| Source code | `src/neural_signal/` |
| Tests | `tests/` |
| Config | `config/` |
| Results | `results/` |
| Notebook | `notebooks/results_analysis.ipynb` |
| README | `README.md` |
| Prompt log | `docs/prompt_engineering_log.md` |

---

## 8. Acceptance Criteria

1. All three models train without NaN/Inf loss
2. Validation MSE decreases or plateaus via early stopping — no divergence
3. `results/clean_noisy_predicted.png` saved with all five overlaid signals
4. `results/comparison_table.csv` contains train/val/test MSE for all three models
5. `uv run ruff check` returns 0 errors
6. `uv run pytest tests/ --cov=src --cov-fail-under=85` passes
7. No secrets committed; `.env-example` present
8. All files ≤ 150 lines
9. `uv.lock` committed and reproducible
10. Prompt engineering log documents all AI-assisted steps

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RNN gradient vanishing | Medium | High | Gradient clipping, verify tanh outputs |
| LSTM overfitting | Medium | Medium | Dropout, early stopping |
| FCN underfitting (no memory) | Low | Medium | Compare MSE; document finding |
| Window size too small for signal | Medium | High | OAT sweep on window_size |
| Coverage below 85% | Medium | High | TDD from the start |
| File exceeds 150 lines | Low | Medium | Ruff enforced, monitor during dev |

---

*Approval required before proceeding to PLAN.md and TODO.md.*
