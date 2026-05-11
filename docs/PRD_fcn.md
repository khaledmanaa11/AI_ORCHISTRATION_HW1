# PRD — Fully Connected Network (FCN) Denoiser

## Overview
The FCN model learns a direct mapping from a windowed, noisy signal concatenated with a one-hot selector vector to the clean signal value at the window centre.

## Input Data
- **Feature tensor**: concatenation of `X_window` (shape `[N, window_size=10]`) and `C` (shape `[N, selector_size=5]`) → `[N, 15]`
- **Target**: clean signal value `y` at window centre → `[N, 1]`
- **Source**: `data/dataset.npz` keys `X_train`, `C_train`, `y_train`

## Output Data
- **Prediction**: scalar denoised value per window → `[N, 1]` float32
- **Checkpoint**: `results/checkpoints/fcn_best.pt` (best val-MSE weights)
- **Training log**: `results/training_log_fcn.csv` (epoch, train_mse, val_mse)

## Architecture
```
Input(15) → Dense(128, ReLU) → Dropout(0.1) → Dense(64, ReLU) → Dense(1)
```
- Hidden sizes: `[128, 64]` (configurable via `config/setup.json → fcn.hidden_sizes`)
- Dropout rate: `0.1` (configurable via `fcn.dropout_rate`)

## Training Setup
| Parameter | Value | Source |
|-----------|-------|--------|
| Optimiser | Adam | `training.learning_rate` |
| Loss | MSELoss | fixed |
| Weight decay | 0.0001 | `training.weight_decay` (FCN only) |
| Max epochs | 200 | `training.max_epochs` |
| Early stopping patience | 10 | `training.early_stopping_patience` |
| Batch size | 64 | `training.batch_size` |

## Validation
- Train/val/test split: 70/15/15 of dataset windows
- Best checkpoint saved when val MSE improves
- Final evaluation on held-out test split (cold load from checkpoint)

## Edge Cases
- Empty `hidden_sizes` list → `ConfigValidationError` raised at load time
- `dropout_rate` outside `[0, 1)` → `ConfigValidationError`
- Missing `X_train` key in `.npz` → `KeyError` from `DataLoaderService`
