# PRD — Vanilla RNN Denoiser

## Overview
The RNN model processes a window of noisy signal samples as a temporal sequence, mapping `(N, window_size, 1)` inputs to a single denoised output value using a recurrent hidden state.

## Input Data
- **Feature tensor**: `X_window` reshaped to `[N, window_size=10, 1]` (sequence of scalar samples)
- **Target**: clean signal value `y` at window centre → `[N, 1]`
- **Source**: `data/dataset.npz` keys `X_train`, `y_train` (selector `C` not used)

## Output Data
- **Prediction**: scalar denoised value per window → `[N, 1]` float32
- **Checkpoint**: `results/checkpoints/rnn_best.pt`
- **Training log**: `results/training_log_rnn.csv`

## Architecture
```
Input(10, 1) → RNN(hidden=64, tanh, layers=1) → last hidden state → Dense(1)
```
- Hidden size: `64` (configurable via `config/setup.json → rnn.hidden_size`)
- Nonlinearity: `"tanh"` (configurable via `rnn.nonlinearity`)
- Num layers: `1` (configurable via `rnn.num_layers`)

## Training Setup
| Parameter | Value | Source |
|-----------|-------|--------|
| Optimiser | Adam | `training.learning_rate` |
| Loss | MSELoss | fixed |
| Weight decay | 0.0 | RNN uses no weight decay |
| Max epochs | 200 | `training.max_epochs` |
| Early stopping patience | 10 | `training.early_stopping_patience` |
| Batch size | 64 | `training.batch_size` |

## Validation
- Uses `seq_train_loader` / `seq_val_loader` / `seq_test_loader` (3D tensors)
- Best checkpoint saved when val MSE improves
- Cold-load evaluation on test split

## Edge Cases
- `hidden_size ≤ 0` → `ConfigValidationError` at load time
- Vanishing gradient risk with long sequences; tanh nonlinearity mitigates somewhat
- With `learning_rate=0.0`, val MSE never improves → early stopping fires within `patience` epochs
