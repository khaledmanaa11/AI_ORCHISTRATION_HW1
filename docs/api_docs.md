# API Reference — neural_signal

All public functionality is accessed through `NeuralSignalSDK`. Service classes are for internal use; their interfaces are documented here for reference.

---

## NeuralSignalSDK (`src/neural_signal/sdk/sdk.py`)

Single entry point for the full pipeline.

### Constructor
```python
NeuralSignalSDK(config_path: Path, rate_limits_path: Path)
```
Loads and validates both JSON config files. Raises `ConfigValidationError` on invalid values.

### Methods

#### `run_all() → RunResult`
Train and evaluate all three models (FCN, RNN, LSTM). Saves checkpoints, training logs, plots, and comparison CSV.

- **Returns**: `RunResult(fcn_eval, rnn_eval, lstm_eval, comparison_df)`
- **Side effects**: creates `.pt` checkpoints, CSV logs, PNG plots, `comparison_table.csv`

```python
sdk = NeuralSignalSDK("config/setup.json", "config/rate_limits.json")
result = sdk.run_all()
print(result.fcn_eval.test_mse)
```

#### `train_model(model_name: str) → TrainingResult`
Train a single model by name.

- **Parameters**: `model_name` — one of `"fcn"`, `"rnn"`, `"lstm"`
- **Returns**: `TrainingResult` with best val MSE, epochs trained, early-stop flag
- **Raises**: `ValueError` if `model_name` is not recognised

```python
result = sdk.train_model("fcn")
print(result.best_val_mse, result.epochs_trained)
```

#### `evaluate_all() → dict[str, EvalResult]`
Load best checkpoints for all models, compute MSE on all splits, save comparison table and all plots.

- **Returns**: `dict[model_name, EvalResult]`
- **Raises**: `FileNotFoundError` if any checkpoint `.pt` file is missing (run `train_model` or `run_all` first)

```python
sdk.train_model("fcn"); sdk.train_model("rnn"); sdk.train_model("lstm")
results = sdk.evaluate_all()
```

#### `get_version() → str`
Return the current package version string (e.g. `"1.00"`).

```python
print(sdk.get_version())   # "1.00"
```

---

## EvalResult (dataclass)
```python
@dataclass
class EvalResult:
    model_name: str
    train_mse: float
    val_mse: float
    test_mse: float
    epochs_trained: int
    stopped_early: bool
```

---

## TrainingResult (dataclass)
```python
@dataclass
class TrainingResult:
    best_val_mse: float
    epochs_trained: int
    stopped_early: bool
    checkpoint_path: str
    train_losses: list[float]
    val_losses: list[float]
```

---

## ConfigManager (`src/neural_signal/shared/config.py`)
```python
ConfigManager(config_path: Path, rate_limits_path: Path).load() → AppConfig
```
Raises `FileNotFoundError` for missing files, `json.JSONDecodeError` for invalid JSON, `ConfigValidationError` for out-of-range values.

---

## Service Layer

### DataLoaderService

```python
DataLoaderService(dataset_path: Path, batch_size: int, seed: int = 42,
                  c_injection_strategy: str = "broadcast").load() → DataBundle
```

| Attribute | Description |
|-----------|-------------|
| `train_loader` / `val_loader` / `test_loader` | FCN batches — shape `(N, 15)` |
| `seq_train_loader` / `seq_val_loader` / `seq_test_loader` | RNN/LSTM batches — shape `(N, 10, 6)` (broadcast C) |
| `X_train`, `C_train`, `y_train` (+ val/test) | Raw numpy arrays |

- **Raises**: `FileNotFoundError` if `.npz` missing; `KeyError` if any of the 9 required arrays absent

### Preprocessor

```python
Preprocessor(params_path: Path)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `fit(X_train: np.ndarray) → ScalerParams` | Compute column-wise mean/std from train only |
| `transform` | `transform(X: np.ndarray) → np.ndarray` | Apply z-score: `(X - mean) / std` |
| `fit_transform` | `fit_transform(X_train) → np.ndarray` | `fit` + `transform` in one call |
| `save_params` | `save_params()` | Write JSON to `params_path` |
| `load_params` | `load_params()` | Load JSON from `params_path` |

- **Raises**: `RuntimeError` if `transform` is called before `fit`

### Trainer

```python
Trainer(cfg: TrainingConfig, checkpoint_path: Path, log_path: Path)
.train(model: nn.Module, train_loader, val_loader,
       weight_decay: float = 0.0) → TrainingResult
```

- Optimizer: Adam with `lr=cfg.learning_rate`, `weight_decay` parameter
- Loss: MSELoss
- Device: `cfg.device` (default `"cpu"`); override via `TORCH_DEVICE` env var
- Gradient clipping applied if `cfg.gradient_clip_norm` is not None
- Saves best checkpoint (lowest val MSE) to `checkpoint_path`
- Writes per-epoch CSV log to `log_path`

### Evaluator

```python
Evaluator(results_dir: Path, device: str = "cpu")
.evaluate(model, model_name, checkpoint_path, train_loader, val_loader, test_loader,
          epochs_trained, stopped_early) → EvalResult
.save_comparison_table(results: list[EvalResult], table_path: Path) → pd.DataFrame
```

- Loads checkpoint if `checkpoint_path` exists before evaluation
- All inference runs under `torch.no_grad()`

### Visualizer

```python
Visualizer(results_dir: Path)
.plot_clean_noisy_predicted(clean, noisy, preds, n_samples=300) → Path
.plot_loss_curves(model_name, train_losses, val_losses) → Path
.plot_mse_comparison(results: list[EvalResult]) → Path
.plot_residuals(model_name, y_true, y_pred) → Path
.plot_pred_vs_actual(predictions: dict[str, tuple[np.ndarray, np.ndarray]]) → Path
```

All methods save a PNG to `results_dir` and return the file path.

### SensitivityAnalyzer

```python
SensitivityAnalyzer(base_config: dict, results_dir: Path)
.run(param_name: str, values: list,
     model_factory: Callable, loader_factory: Callable) → list[SensitivityResult]
.save_results(results: list[SensitivityResult]) → Path          # → sensitivity_results.csv
.plot_line_chart(param_name: str, results) → Path               # → sensitivity_<param>.png
.plot_heatmap(results) → Path                                    # → sensitivity_heatmap.png
.make_simple_loaders(n_train, n_val, n_features, batch_size) → (DataLoader, DataLoader)
```

---

## CLI (`uv run python -m neural_signal.main`)

```
usage: main.py [-h] [--config PATH] [--rate-limits PATH]
               [--model {fcn,rnn,lstm,all}] [--verbose]

optional arguments:
  --config       Path to setup.json (default: config/setup.json)
  --rate-limits  Path to rate_limits.json (default: config/rate_limits.json)
  --model        Which model to train: fcn | rnn | lstm | all (default: all)
  --verbose      Print per-epoch progress
```
