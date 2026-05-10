# HW1 — Noisy Sine Signal Dataset + Neural Signal Regression

A two-package Python project that:
1. **signal_dataset** — Generates a labeled dataset from four pure sine waves, each corrupted
   by Gaussian + burst noise, windowed with a one-hot context selector.
2. **neural_signal** — Trains and evaluates three neural network architectures (FCN, RNN, LSTM)
   on that dataset for conditional source-separation / denoising regression.

---

## Training Results

| Model | Train MSE | Val MSE | Test MSE | Epochs | Early Stop |
|-------|-----------|---------|----------|--------|------------|
| **FCN** | 0.0543 | 0.0801 | **0.0817** | 76 | No |
| **LSTM** | 0.7721 | 0.8137 | 0.8173 | 28 | No |
| **RNN** | 0.7783 | 0.8254 | 0.8245 | 26 | No |

FCN significantly outperforms the sequence models on this task because the one-hot conditioning
vector C already encodes all the signal-selection context the model needs — removing the advantage
of temporal gating.

---

## Signals

| ID | Formula | Amplitude | Frequency (Hz) | Phase (rad) |
|----|---------|-----------|----------------|-------------|
| s1 | 2.0 · sin(2π · 5 · t) | 2.0 | 5 | 0 |
| s2 | 1.5 · sin(2π · 15 · t + π/4) | 1.5 | 15 | π/4 |
| s3 | 0.8 · sin(2π · 50 · t) | 0.8 | 50 | 0 |
| s4 | 0.3 · sin(2π · 100 · t) | 0.3 | 100 | 0 |
| s5 | s1 + s2 + s3 + s4 (composite) | — | — | — |

---

## Noise Model

For each signal `si`, the noisy version is:

```
s_noisy(t) = (A_i + N_amp(t)) · sin(2π · f_i · t + φ_i + N_phase(t))
```

- `N_amp = gaussian_amp + burst_amp` — amplitude perturbation
- `N_phase = gaussian_phase + burst_phase` — phase perturbation
- Gaussian: per-sample white noise with configurable σ per signal
- Burst: random-duration spikes (probability, duration, magnitude all configurable)

---

## Task Formulation

```
input  = concat([x_window, C])   shape: (W + 5,) = (15,)   ← FCN
input  = x_window.reshape(W, 1)  shape: (10, 1)             ← RNN / LSTM
target = scalar clean value of the C-selected signal at the window centre
```

`x_window` is a length-W slice of `s5_noisy`; `C` is a one-hot vector selecting which of the
five clean components to recover. The model learns conditional source separation.

---

## Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Generate dataset
uv run python -m signal_dataset

# 3. Train all three models & save results
uv run python -m neural_signal

# 4. Run tests with coverage
uv run pytest tests/

# 5. Lint (must be zero errors)
uv run ruff check
```

---

## Project Structure

```
HW1/
├── src/
│   ├── signal_dataset/               # Phase 1 — dataset generation
│   │   ├── sdk/sdk.py                # DatasetSDK entry point
│   │   ├── services/
│   │   │   ├── signal_generator.py   # Pure sine wave generation
│   │   │   ├── noise_injector.py     # Gaussian + burst noise
│   │   │   ├── windower.py           # Sliding window + one-hot C
│   │   │   └── dataset_builder.py    # Train/val/test splits → .npz
│   │   └── shared/
│   │       ├── config.py
│   │       ├── gatekeeper.py
│   │       └── version.py
│   └── neural_signal/                # Phase 2 — neural regression
│       ├── sdk/sdk.py                # NeuralSignalSDK entry point
│       ├── models/
│       │   ├── fcn.py                # Fully-connected (Dense 128→64→1)
│       │   ├── rnn.py                # Vanilla RNN (hidden=64, tanh)
│       │   └── lstm.py               # LSTM (hidden=64, dense 32→1)
│       ├── services/
│       │   ├── data_loader.py        # DataLoaderService + DataBundle
│       │   ├── preprocessor.py       # Z-score normalisation
│       │   ├── trainer.py            # Adam + MSE + early stopping
│       │   ├── evaluator.py          # MSE evaluation + CSV table
│       │   └── visualizer.py         # All result plots (PNG)
│       └── shared/
│           ├── config.py             # ConfigManager + all dataclasses
│           ├── gatekeeper.py         # ApiGatekeeper (rate limits)
│           └── version.py
├── tests/
│   ├── conftest.py                   # Shared fixtures (both packages)
│   ├── unit/                         # 340+ unit tests (TDD)
│   └── integration/                  # End-to-end pipeline tests
├── config/
│   ├── setup.json                    # All parameters (dataset + training)
│   └── rate_limits.json              # API gatekeeper rate limits
├── data/
│   ├── dataset.npz                   # Generated dataset (X, C, y splits)
│   └── signals_raw.npz               # Raw clean + noisy signal vectors
├── results/                          # All generated plots + tables
│   ├── clean_noisy_predicted.png
│   ├── mse_comparison.png
│   ├── loss_curves_{fcn,rnn,lstm}.png
│   ├── residuals_{fcn,rnn,lstm}.png
│   ├── pred_vs_actual.png
│   ├── comparison_table.csv
│   └── scaler_params.json
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   └── TODO.md
├── pyproject.toml                    # All dependencies (uv)
└── uv.lock
```

---

## Architecture

### FCN (Fully Connected Network)
- Input: `(N, 15)` — 10-sample window concatenated with 5-dim one-hot C
- Hidden: Dense(128) → ReLU → Dropout(0.1) → Dense(64) → ReLU → Dropout(0.1)
- Output: Dense(1)

### RNN (Vanilla Recurrent Network)
- Input: `(N, 10, 1)` — window reshaped as sequence, many-to-one
- Hidden: RNN(hidden=64, tanh, batch_first=True)
- Output: last hidden state → Linear(64, 1)

### LSTM (Long Short-Term Memory)
- Input: `(N, 10, 1)` — sequence, many-to-one
- Hidden: LSTM(hidden=64, batch_first=True)
- Output: last hidden state → Dense(32) → ReLU → Dense(1)

### Training Setup
- Optimiser: Adam (lr=0.001, weight_decay=1e-4)
- Loss: MSE
- Batch size: 64
- Max epochs: 200
- Early stopping: patience=10 on val MSE
- Normalisation: Z-score (fit on train only, params saved to `results/scaler_params.json`)

---

## Configuration

All parameters live in `config/setup.json` — no hardcoded values in source code.

| Section | Key | Default |
|---------|-----|---------|
| `dataset` | `duration_sec` | 10 |
| `dataset` | `sample_rate_hz` | 1000 |
| `dataset` | `window_size` | 10 |
| `training` | `batch_size` | 64 |
| `training` | `learning_rate` | 0.001 |
| `training` | `max_epochs` | 200 |
| `training` | `early_stopping_patience` | 10 |
| `fcn` | `hidden_sizes` | [128, 64] |
| `rnn` | `hidden_size` | 64 |
| `lstm` | `hidden_size` | 64 |

---

## Testing

```bash
uv run pytest tests/          # all tests with coverage report
uv run pytest tests/unit/     # unit tests only
uv run pytest tests/integration/
```

| Metric | Value |
|--------|-------|
| Tests | 377 passed |
| Coverage | 98.37% |
| Threshold | 85% (enforced — build fails below) |

---

## Linting

```bash
uv run ruff check
```

Zero errors enforced across `src/` and `tests/`. Configuration in `pyproject.toml`:
- `select = ["E","F","W","I","N","UP","B","C4","SIM"]`
- `ignore = ["E501"]`
- `line-length = 100`

---

## Results

### Clean vs Noisy vs Predicted
![clean_noisy_predicted](results/clean_noisy_predicted.png)

### MSE Comparison (All Models)
![mse_comparison](results/mse_comparison.png)

### Training Loss Curves — FCN
![loss_fcn](results/loss_curves_fcn.png)

### Training Loss Curves — RNN
![loss_rnn](results/loss_curves_rnn.png)

### Training Loss Curves — LSTM
![loss_lstm](results/loss_curves_lstm.png)

### Residuals — FCN
![residuals_fcn](results/residuals_fcn.png)

### Predicted vs Actual
![pred_vs_actual](results/pred_vs_actual.png)

---

## Design Rationale — Why These Four Signals?

| Signal | Role | Architecture Stress |
|--------|------|---------------------|
| **s1** — 5 Hz, A=2.0 | Dominant slow trend | FCN cannot distinguish trend without temporal context |
| **s2** — 15 Hz, A=1.5, φ=π/4 | Phase-shifted mid-band | Tests whether the model memorises absolute phase vs. infers it from context |
| **s3** — 50 Hz, A=0.8 | High-frequency oscillation | Exceeds RNN memory horizon; vanishing gradients prevent phase tracking |
| **s4** — 100 Hz, A=0.3 | Near-Nyquist component | Only 10 samples/cycle; forces separation of signal from noise |

---

## Author

Khaled Mnaa — khaled.mnaa43@gmail.com
