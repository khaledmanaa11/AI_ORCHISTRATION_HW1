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

#### `run_all() → dict[str, EvalResult]`
Train and evaluate all three models (FCN, RNN, LSTM). Saves checkpoints, training logs, plots, and comparison CSV. Returns a dict keyed by model name.

#### `train_model(model_name: str) → TrainingResult`
Train a single model. `model_name` must be `"fcn"`, `"rnn"`, or `"lstm"`.

#### `evaluate_all() → dict[str, EvalResult]`
Load best checkpoints, compute MSE on all splits, save comparison table and all plots.

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
DataLoaderService(dataset_path: Path, batch_size: int, seed: int = 42).load() → DataBundle
```
- Raises `FileNotFoundError` if `.npz` missing
- Raises `KeyError` if required arrays absent
- FCN loaders: shape `(N, 15)` — window + selector concat
- RNN/LSTM loaders: shape `(N, 10, 1)` — sequential window

### Preprocessor
```python
Preprocessor(params_path: Path)
.fit_transform(X: np.ndarray) → np.ndarray   # z-score normalise, fit on train
.transform(X: np.ndarray) → np.ndarray        # apply saved params
.save_params()                                  # write JSON to params_path
```

### Trainer
```python
Trainer(cfg: TrainingConfig, checkpoint_path: Path, log_path: Path)
.train(model, train_loader, val_loader, weight_decay=0.0) → TrainingResult
```
- Device from `cfg.device` (default `"cpu"`)
- Gradient clipping if `cfg.gradient_clip_norm` is not None

### Evaluator
```python
Evaluator(results_dir: Path, device: str = "cpu")
.evaluate(model, model_name, checkpoint_path, train_loader, val_loader, test_loader,
          epochs_trained, stopped_early) → EvalResult
.save_comparison_table(results: list[EvalResult], table_path: Path) → pd.DataFrame
```

### Visualizer
```python
Visualizer(results_dir: Path)
.plot_clean_noisy_predicted(clean, noisy, preds, n_samples=300) → Path
.plot_loss_curves(model_name, train_losses, val_losses) → Path
.plot_mse_comparison(results: list[EvalResult]) → Path
.plot_residuals(model_name, y_true, y_pred) → Path
.plot_pred_vs_actual(predictions: dict[str, tuple[np.ndarray, np.ndarray]]) → Path
```

### SensitivityAnalyzer
```python
SensitivityAnalyzer(base_config: dict, results_dir: Path)
.run(param_name, values, model_factory, loader_factory) → list[SensitivityResult]
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
