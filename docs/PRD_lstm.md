# PRD — LSTM Denoiser

## Overview
The LSTM model extends the RNN with gated memory cells, giving it better long-range temporal modelling. It maps a `(N, window_size, 1)` sequence to a denoised scalar via a dense head.

## Input Data
- **Feature tensor**: `X_window` reshaped to `[N, window_size=10, 1]`
- **Target**: clean signal value `y` at window centre → `[N, 1]`
- **Source**: `data/dataset.npz` keys `X_train`, `y_train`

## Output Data
- **Prediction**: scalar denoised value per window → `[N, 1]` float32
- **Checkpoint**: `results/checkpoints/lstm_best.pt`
- **Training log**: `results/training_log_lstm.csv`

## Architecture
```
Input(10, 1) → LSTM(hidden=64, layers=1) → last hidden state
             → Dense(32, ReLU) → Dense(1)
```
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
