# PRD — LSTM Denoiser

## Overview
The LSTM model extends the RNN with gated memory cells, giving it better long-range temporal modelling. It maps a `(N, window_size, 1)` sequence to a denoised scalar via a dense head.

## Input Data
- **Feature tensor**: `X_window` broadcast-concatenated with `C` → `[N, window_size=10, 6]`
  - `X_window` shape `(N, 10)` reshaped to `(N, 10, 1)`; `C` shape `(N, 5)` broadcast to `(N, 10, 5)`; concatenated on feature axis → `(N, 10, 6)`
- **C injection strategy**: `"broadcast"` — one-hot signal selector (5 elements) appended to every time-step, giving `input_size = 6`
- **Target**: clean signal value `y` at window centre → `[N, 1]`
- **Source**: `data/dataset.npz` keys `X_train`, `C_train`, `y_train`
- **Config key**: `config/setup.json → lstm.input_size = 6`, `training.c_injection_strategy = "broadcast"`

## Output Data
- **Prediction**: scalar denoised value per window → `[N, 1]` float32
- **Checkpoint**: `results/checkpoints/lstm_best.pt`
- **Training log**: `results/training_log_lstm.csv`

## Architecture
```
Input(10, 6) → LSTM(input_size=6, hidden=64, layers=1) → last output
             → Dense(32, ReLU) → Dense(1)
```
- Input size: `6` = 1 window feature + 5 C features per step (from `config/setup.json → lstm.input_size`)
- Hidden size: `64` (configurable via `config/setup.json → lstm.hidden_size`)
- Num layers: `1` (configurable via `lstm.num_layers`)
- Dense hidden size: `32` (configurable via `lstm.dense_hidden_size`)

## Training Setup
| Parameter | Value | Source |
|-----------|-------|--------|
| Optimiser | Adam | `training.learning_rate` |
| Loss | MSELoss | fixed |
| Weight decay | 0.0 | LSTM uses no weight decay |
| Max epochs | 200 | `training.max_epochs` |
| Early stopping patience | 10 | `training.early_stopping_patience` |
| Batch size | 64 | `training.batch_size` |
| Gradient clipping | null | `training.gradient_clip_norm` (optional) |

## Validation
- Uses `seq_train_loader` / `seq_val_loader` / `seq_test_loader`
- Best checkpoint saved when val MSE improves
- Cold-load evaluation on test split

## Edge Cases
- `hidden_size ≤ 0` → `ConfigValidationError`
- `dense_hidden_size` must be positive; validated downstream by PyTorch layer construction
- Gradient clipping enabled via `training.gradient_clip_norm` if not null
