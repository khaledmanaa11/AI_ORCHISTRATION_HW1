# TODO — Neural Network Signal Regression (FCN · RNN · LSTM)

**Version:** 1.00
**Date:** 2026-05-10
**Status:** Not started — awaiting doc approval

Legend: `[ ]` = pending · `[~]` = in progress · `[x]` = done

---

## Phase 0 — Documentation & Approval

- [ ] Read PRD.md in full and verify all three model architectures are correctly specified
- [ ] Verify FCN spec: Dense(128,ReLU) → Dropout(0.1) → Dense(64,ReLU) → Dropout(0.1) → Dense(1,Linear)
- [ ] Verify FCN L2 weight decay 1e-4 stated in PRD.md
- [ ] Verify RNN spec: hidden_size=64, nonlinearity=tanh, many-to-one (last hidden state)
- [ ] Verify LSTM spec: hidden_size=64, Dense(32,ReLU), Dense(1), many-to-one (last output)
- [ ] Verify optimizer: Adam, lr=0.001 stated in PRD.md
- [ ] Verify loss: MSE stated in PRD.md
- [ ] Verify batch size: 64 stated in PRD.md
- [ ] Verify validation split: 20% stated in PRD.md
- [ ] Verify early stopping stated in PRD.md
- [ ] Verify mandatory visualization: "Clean vs. Noisy vs. Predicted" in PRD.md
- [ ] Verify all 5 deliverable plot files listed in PRD.md
- [ ] Verify per-algorithm PRDs listed: PRD_fcn.md, PRD_rnn.md, PRD_lstm.md
- [ ] Verify acceptance criteria count is 10 in PRD.md
- [ ] Read PLAN.md in full and verify repository structure matches PRD scope
- [ ] Verify all model files listed in PLAN.md: fcn.py, rnn.py, lstm.py
- [ ] Verify all service files listed in PLAN.md: data_loader, preprocessor, trainer, evaluator, visualizer
- [ ] Verify DataLoaderService input encoding described for FCN and RNN/LSTM
- [ ] Verify config JSON snippet for training, fcn, rnn, lstm, output in PLAN.md
- [ ] Verify file size budget table — all entries ≤ 150 lines
- [ ] Verify testing strategy table lists all 12 test files
- [ ] Verify 10 visualisation files listed in PLAN.md
- [ ] Read TODO.md (this file) and confirm all phases are present
- [ ] Submit PRD.md to supervisor for approval
- [ ] Submit PLAN.md to supervisor for approval
- [ ] Submit TODO.md to supervisor for approval
- [ ] Record approval date and version for PRD.md
- [ ] Record approval date and version for PLAN.md
- [ ] Record approval date and version for TODO.md

---

## Phase 1 — Environment Setup

- [ ] Verify Python 3.10+ installed: `python --version`
- [ ] Verify uv installed: `uv --version`
- [ ] Verify ruff available: `ruff --version`
- [ ] Run `uv add torch` — add PyTorch to pyproject.toml
- [ ] Verify torch version ≥ 2.0 installed
- [ ] Run `uv add numpy` — verify added to pyproject.toml
- [ ] Run `uv add pandas` — verify added
- [ ] Run `uv add matplotlib` — verify added
- [ ] Run `uv add seaborn` — verify added
- [ ] Run `uv add jupyter` — verify added
- [ ] Run `uv add scipy` — verify added
- [ ] Run `uv add --dev pytest` — verify added to dev deps
- [ ] Run `uv add --dev pytest-cov` — verify added
- [ ] Run `uv add --dev ruff` — verify added
- [ ] Run `uv add --dev pytest-mock` — verify added
- [ ] Run `uv lock` and verify `uv.lock` updated
- [ ] Run `uv sync` — verify all packages install without errors
- [ ] Confirm no `requirements.txt` present
- [ ] Confirm no `pip install` commands used anywhere
- [ ] Verify `import torch` works in `uv run python -c "import torch; print(torch.__version__)"`
- [ ] Verify `import torch.nn` works
- [ ] Verify `import torch.optim` works
- [ ] Verify `import torch.utils.data` works
- [ ] Confirm CUDA availability check in config (graceful CPU fallback)

---

## Phase 2 — Folder Structure Creation

- [ ] Create `src/neural_signal/` directory
- [ ] Create `src/neural_signal/sdk/` directory
- [ ] Create `src/neural_signal/models/` directory
- [ ] Create `src/neural_signal/services/` directory
- [ ] Create `src/neural_signal/shared/` directory
- [ ] Create `tests/unit/` directory (if not present)
- [ ] Create `tests/integration/` directory (if not present)
- [ ] Create `results/checkpoints/` directory
- [ ] Create `results/sensitivity/` directory
- [ ] Create `notebooks/` directory (if not present)
- [ ] Verify full tree matches PLAN.md exactly
- [ ] Verify `data/dataset.npz` exists (generated in Phase 1)

---

## Phase 3 — pyproject.toml Configuration

- [ ] Set `[project] name = "neural-signal"`
- [ ] Set `[project] version = "1.0.0"`
- [ ] Set `[project] requires-python = ">=3.10"`
- [ ] Add `torch` to `[project] dependencies`
- [ ] Add `numpy` to `[project] dependencies`
- [ ] Add `pandas` to `[project] dependencies`
- [ ] Add `matplotlib` to `[project] dependencies`
- [ ] Add `seaborn` to `[project] dependencies`
- [ ] Add `scipy` to `[project] dependencies`
- [ ] Add `jupyter` to `[project] dependencies`
- [ ] Add `pytest` to `[project.optional-dependencies] dev`
- [ ] Add `pytest-cov` to dev deps
- [ ] Add `ruff` to dev deps
- [ ] Add `pytest-mock` to dev deps
- [ ] Add `[tool.ruff]` section with `line-length = 100`
- [ ] Add `[tool.ruff]` section with `target-version = "py310"`
- [ ] Add `[tool.ruff.lint] select = ["E","F","W","I","N","UP","B","C4","SIM"]`
- [ ] Add `[tool.ruff.lint] ignore = ["E501"]`
- [ ] Add `[tool.pytest.ini_options] testpaths = ["tests"]`
- [ ] Add `[tool.pytest.ini_options] addopts = "--cov=src --cov-report=term-missing"`
- [ ] Add `[tool.coverage.run] source = ["src"]`
- [ ] Add `[tool.coverage.run] omit = ["src/*/main.py", "*/tests/*"]`
- [ ] Add `[tool.coverage.report] fail_under = 85`
- [ ] Add `[build-system]` table
- [ ] Add `[project.scripts] neural-signal = "neural_signal.main:main"`
- [ ] Verify `uv run ruff check pyproject.toml` passes
- [ ] Verify `uv sync` still works after all edits

---

## Phase 4 — Configuration Files

- [ ] Open `config/setup.json` for editing
- [ ] Add `"training"` block to setup.json
- [ ] Add `"batch_size": 64` to training block
- [ ] Add `"learning_rate": 0.001` to training block
- [ ] Add `"weight_decay": 0.0001` to training block
- [ ] Add `"max_epochs": 200` to training block
- [ ] Add `"early_stopping_patience": 10` to training block
- [ ] Add `"val_split": 0.20` to training block
- [ ] Add `"random_seed": 42` to training block
- [ ] Add `"fcn"` block to setup.json
- [ ] Add `"hidden_sizes": [128, 64]` to fcn block
- [ ] Add `"dropout_rate": 0.1` to fcn block
- [ ] Add `"output_size": 1` to fcn block
- [ ] Add `"rnn"` block to setup.json
- [ ] Add `"hidden_size": 64` to rnn block
- [ ] Add `"nonlinearity": "tanh"` to rnn block
- [ ] Add `"num_layers": 1` to rnn block
- [ ] Add `"output_size": 1` to rnn block
- [ ] Add `"lstm"` block to setup.json
- [ ] Add `"hidden_size": 64` to lstm block
- [ ] Add `"num_layers": 1` to lstm block
- [ ] Add `"dense_hidden_size": 32` to lstm block
- [ ] Add `"output_size": 1` to lstm block
- [ ] Add or update `"output"` block in setup.json
- [ ] Add `"results_dir": "results/"` to output block
- [ ] Add `"checkpoint_dir": "results/checkpoints/"` to output block
- [ ] Add `"fcn_checkpoint": "results/checkpoints/fcn_best.pt"` to output block
- [ ] Add `"rnn_checkpoint": "results/checkpoints/rnn_best.pt"` to output block
- [ ] Add `"lstm_checkpoint": "results/checkpoints/lstm_best.pt"` to output block
- [ ] Add `"comparison_table_path": "results/comparison_table.csv"` to output block
- [ ] Add `"scaler_params_path": "results/scaler_params.json"` to output block
- [ ] Validate `setup.json` is valid JSON
- [ ] Confirm `config/rate_limits.json` exists with default service block
- [ ] Verify no hyperparameter literals appear in Python source (grep check)

---

## Phase 5 — .gitignore and .env-example

- [ ] Verify `.gitignore` contains `.env`
- [ ] Verify `.gitignore` contains `*.key`
- [ ] Verify `.gitignore` contains `*.pem`
- [ ] Verify `.gitignore` contains `credentials.json`
- [ ] Add `results/checkpoints/` to `.gitignore` (large model files)
- [ ] Verify `__pycache__/` in `.gitignore`
- [ ] Verify `*.pyc` in `.gitignore`
- [ ] Verify `.pytest_cache/` in `.gitignore`
- [ ] Verify `.coverage` in `.gitignore`
- [ ] Verify `htmlcov/` in `.gitignore`
- [ ] Verify `.ruff_cache/` in `.gitignore`
- [ ] Verify `.venv/` in `.gitignore`
- [ ] Verify `.env-example` exists and has dummy values
- [ ] Add `TORCH_DEVICE=cpu` to `.env-example`
- [ ] Add `LOG_LEVEL=INFO` to `.env-example`
- [ ] Confirm `.env` is not tracked by git

---

## Phase 6 — `shared/version.py`

- [ ] Create `src/neural_signal/shared/version.py`
- [ ] Add module docstring explaining version tracking purpose
- [ ] Define `__version__ = "1.00"`
- [ ] Define `VERSION_MAJOR = 1`
- [ ] Define `VERSION_MINOR = 0`
- [ ] Define `BUILD_DATE = "2026-05-10"`
- [ ] Define `get_version() -> str` returning `__version__`
- [ ] Add docstring to `get_version()`
- [ ] Verify file ≤ 150 lines
- [ ] Verify `uv run ruff check src/neural_signal/shared/version.py` → 0 errors

---

## Phase 7 — `constants.py`

- [ ] Create `src/neural_signal/constants.py`
- [ ] Add module docstring
- [ ] Define `MODEL_FCN = "fcn"`
- [ ] Define `MODEL_RNN = "rnn"`
- [ ] Define `MODEL_LSTM = "lstm"`
- [ ] Define `MODEL_NAMES = [MODEL_FCN, MODEL_RNN, MODEL_LSTM]`
- [ ] Define `INPUT_SIZE_FCN = 15` — window(10) + selector(5)
- [ ] Define `WINDOW_SIZE = 10`
- [ ] Define `SELECTOR_SIZE = 5`
- [ ] Define `DATASET_KEY_X_TRAIN = "X_train"`
- [ ] Define `DATASET_KEY_X_VAL = "X_val"`
- [ ] Define `DATASET_KEY_X_TEST = "X_test"`
- [ ] Define `DATASET_KEY_C_TRAIN = "C_train"`
- [ ] Define `DATASET_KEY_C_VAL = "C_val"`
- [ ] Define `DATASET_KEY_C_TEST = "C_test"`
- [ ] Define `DATASET_KEY_Y_TRAIN = "y_train"`
- [ ] Define `DATASET_KEY_Y_VAL = "y_val"`
- [ ] Define `DATASET_KEY_Y_TEST = "y_test"`
- [ ] Define `RESULT_COL_MODEL = "model"`
- [ ] Define `RESULT_COL_TRAIN_MSE = "train_mse"`
- [ ] Define `RESULT_COL_VAL_MSE = "val_mse"`
- [ ] Define `RESULT_COL_TEST_MSE = "test_mse"`
- [ ] Define `RESULT_COL_EPOCHS = "epochs_trained"`
- [ ] Define `RESULT_COL_STOPPED_EARLY = "stopped_early"`
- [ ] Define `PLOT_COLOR_CLEAN = "green"`
- [ ] Define `PLOT_COLOR_NOISY = "grey"`
- [ ] Define `PLOT_COLOR_FCN = "red"`
- [ ] Define `PLOT_COLOR_RNN = "orange"`
- [ ] Define `PLOT_COLOR_LSTM = "purple"`
- [ ] Verify file ≤ 150 lines
- [ ] Verify `uv run ruff check src/neural_signal/constants.py` → 0 errors

---

## Phase 8 — `__init__.py` Files

- [ ] Create `src/neural_signal/__init__.py`
- [ ] Add module docstring to `__init__.py`
- [ ] Import `__version__` from `shared.version`
- [ ] Import `NeuralSignalSDK` from `sdk.sdk`
- [ ] Define `__all__ = ["NeuralSignalSDK", "__version__"]`
- [ ] Verify file ≤ 150 lines
- [ ] Verify ruff passes
- [ ] Create `src/neural_signal/sdk/__init__.py` (empty with docstring)
- [ ] Create `src/neural_signal/models/__init__.py` (empty with docstring)
- [ ] Create `src/neural_signal/services/__init__.py` (empty with docstring)
- [ ] Create `src/neural_signal/shared/__init__.py` (empty with docstring)
- [ ] Create `tests/unit/__init__.py` (empty)
- [ ] Create `tests/integration/__init__.py` (empty)

---

## Phase 9 — `tests/conftest.py` (Shared Fixtures)

- [ ] Create `tests/conftest.py`
- [ ] Add module docstring explaining shared fixtures
- [ ] Import `pytest`, `numpy`, `torch`, `pathlib.Path`
- [ ] Define `config_path` fixture → path to `config/setup.json`
- [ ] Define `rate_limits_path` fixture → path to `config/rate_limits.json`
- [ ] Define `mock_training_config` fixture — minimal valid training config dict
- [ ] Define `mock_fcn_config` fixture — hidden_sizes, dropout_rate, output_size
- [ ] Define `mock_rnn_config` fixture — hidden_size=64, nonlinearity=tanh, num_layers=1
- [ ] Define `mock_lstm_config` fixture — hidden_size=64, dense_hidden_size=32
- [ ] Define `tiny_X_train` fixture — shape (100, 10) random float tensor
- [ ] Define `tiny_C_train` fixture — shape (100, 5) valid one-hot tensor
- [ ] Define `tiny_y_train` fixture — shape (100, 1) random float tensor
- [ ] Define `tiny_x_input_fcn` fixture — shape (100, 15): concat(X, C)
- [ ] Define `tiny_x_seq_rnn` fixture — shape (100, 10, 1): X reshaped
- [ ] Define `tmp_results_dir` fixture using `tmp_path`
- [ ] Define `tmp_checkpoint_path` fixture using `tmp_path / "model_best.pt"`
- [ ] Define `batch_size_fixture` returning 16 (small for tests)
- [ ] Verify conftest.py ≤ 150 lines
- [ ] Verify `uv run ruff check tests/conftest.py` → 0 errors

---

## Phase 10 — `shared/config.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_config.py`
- [ ] Add module docstring
- [ ] Write `test_config_loads_setup_json_without_error`
- [ ] Write `test_config_training_batch_size_equals_64`
- [ ] Write `test_config_training_learning_rate_equals_0_001`
- [ ] Write `test_config_training_weight_decay_equals_1e_4`
- [ ] Write `test_config_training_max_epochs_equals_200`
- [ ] Write `test_config_training_early_stopping_patience_equals_10`
- [ ] Write `test_config_training_val_split_equals_0_20`
- [ ] Write `test_config_training_random_seed_equals_42`
- [ ] Write `test_config_fcn_hidden_sizes_equals_128_64`
- [ ] Write `test_config_fcn_dropout_rate_equals_0_1`
- [ ] Write `test_config_fcn_output_size_equals_1`
- [ ] Write `test_config_rnn_hidden_size_equals_64`
- [ ] Write `test_config_rnn_nonlinearity_equals_tanh`
- [ ] Write `test_config_rnn_num_layers_equals_1`
- [ ] Write `test_config_rnn_output_size_equals_1`
- [ ] Write `test_config_lstm_hidden_size_equals_64`
- [ ] Write `test_config_lstm_dense_hidden_size_equals_32`
- [ ] Write `test_config_lstm_num_layers_equals_1`
- [ ] Write `test_config_lstm_output_size_equals_1`
- [ ] Write `test_config_output_results_dir_is_string`
- [ ] Write `test_config_output_checkpoint_dir_is_string`
- [ ] Write `test_config_output_fcn_checkpoint_contains_fcn`
- [ ] Write `test_config_output_rnn_checkpoint_contains_rnn`
- [ ] Write `test_config_output_lstm_checkpoint_contains_lstm`
- [ ] Write `test_config_raises_on_missing_file`
- [ ] Write `test_config_raises_on_invalid_json`
- [ ] Write `test_config_raises_on_negative_learning_rate`
- [ ] Write `test_config_raises_on_zero_batch_size`
- [ ] Write `test_config_raises_on_max_epochs_less_than_1`
- [ ] Write `test_config_raises_on_patience_less_than_1`
- [ ] Write `test_config_raises_on_val_split_not_in_0_1`
- [ ] Write `test_config_raises_on_empty_fcn_hidden_sizes`
- [ ] Write `test_config_raises_on_negative_dropout_rate`
- [ ] Write `test_config_raises_on_dropout_rate_above_1`
- [ ] Write `test_config_raises_on_rnn_hidden_size_zero`
- [ ] Write `test_config_raises_on_lstm_hidden_size_zero`
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/shared/config.py`
- [ ] Add module docstring
- [ ] Import `json`, `pathlib`, `dataclasses`, `typing`
- [ ] Define `TrainingConfig` dataclass: batch_size, learning_rate, weight_decay, max_epochs, early_stopping_patience, val_split, random_seed
- [ ] Add docstring to `TrainingConfig`
- [ ] Define `FCNConfig` dataclass: hidden_sizes, dropout_rate, output_size
- [ ] Add docstring to `FCNConfig`
- [ ] Define `RNNConfig` dataclass: hidden_size, nonlinearity, num_layers, output_size
- [ ] Add docstring to `RNNConfig`
- [ ] Define `LSTMConfig` dataclass: hidden_size, num_layers, dense_hidden_size, output_size
- [ ] Add docstring to `LSTMConfig`
- [ ] Define `OutputConfig` dataclass: results_dir, checkpoint_dir, fcn_checkpoint, rnn_checkpoint, lstm_checkpoint, comparison_table_path, scaler_params_path
- [ ] Add docstring to `OutputConfig`
- [ ] Define `AppConfig` dataclass: training, fcn, rnn, lstm, output
- [ ] Add docstring to `AppConfig`
- [ ] Define `ConfigValidationError(Exception)`
- [ ] Define `ConfigManager` class with docstring
- [ ] Implement `__init__(self, config_path, rate_limits_path)`
- [ ] Implement `load() -> AppConfig`
- [ ] Implement `_load_json(path) -> dict`
- [ ] Implement `_parse_training(raw) -> TrainingConfig`
- [ ] Implement `_parse_fcn(raw) -> FCNConfig`
- [ ] Implement `_parse_rnn(raw) -> RNNConfig`
- [ ] Implement `_parse_lstm(raw) -> LSTMConfig`
- [ ] Implement `_parse_output(raw) -> OutputConfig`
- [ ] Implement `_validate(config: AppConfig)` raising `ConfigValidationError`
- [ ] Validate learning_rate > 0
- [ ] Validate batch_size ≥ 1
- [ ] Validate max_epochs ≥ 1
- [ ] Validate patience ≥ 1
- [ ] Validate val_split in (0, 1)
- [ ] Validate dropout_rate in [0, 1)
- [ ] Validate fcn hidden_sizes non-empty
- [ ] Validate rnn hidden_size > 0
- [ ] Validate lstm hidden_size > 0
- [ ] Verify file ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Extract validation helpers into private methods
- [ ] Confirm all error messages include field name and bad value
- [ ] Run `uv run ruff check src/neural_signal/shared/config.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 11 — `shared/gatekeeper.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_gatekeeper.py`
- [ ] Add module docstring
- [ ] Write `test_gatekeeper_instantiates_with_valid_config`
- [ ] Write `test_gatekeeper_execute_calls_function`
- [ ] Write `test_gatekeeper_execute_returns_result`
- [ ] Write `test_gatekeeper_execute_logs_call`
- [ ] Write `test_gatekeeper_queue_status_returns_object`
- [ ] Write `test_gatekeeper_queue_depth_starts_at_0`
- [ ] Write `test_gatekeeper_processed_count_grows_after_execute`
- [ ] Write `test_gatekeeper_failed_count_grows_on_exception`
- [ ] Write `test_gatekeeper_get_call_log_returns_list`
- [ ] Write `test_gatekeeper_call_log_entry_has_timestamp`
- [ ] Write `test_gatekeeper_call_log_entry_has_function_name`
- [ ] Write `test_gatekeeper_call_log_entry_has_status_success`
- [ ] Write `test_gatekeeper_call_log_entry_has_status_failure`
- [ ] Write `test_gatekeeper_retries_on_exception`
- [ ] Write `test_gatekeeper_raises_after_max_retries`
- [ ] Write `test_gatekeeper_raises_when_queue_full`
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/shared/gatekeeper.py`
- [ ] Add module docstring
- [ ] Import `time`, `queue`, `dataclasses`, `typing`, `logging`
- [ ] Define `RateLimitConfig` dataclass
- [ ] Define `QueueStatus` dataclass: depth, processed, failed
- [ ] Define `CallLogEntry` dataclass: timestamp, function_name, status, error_message
- [ ] Define `GatekeeperQueueFullError(Exception)`
- [ ] Define `GatekeeperMaxRetriesError(Exception)`
- [ ] Define `ApiGatekeeper` class with docstring
- [ ] Implement `__init__(self, config: RateLimitConfig)`
- [ ] Implement `execute(self, api_call, *args, **kwargs)`
- [ ] Implement retry loop up to `max_retries`
- [ ] Implement `get_queue_status() -> QueueStatus`
- [ ] Implement `get_call_log() -> list[CallLogEntry]`
- [ ] Implement private `_log_call(name, status, error)`
- [ ] Verify file ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Extract retry logic into `_execute_with_retry`
- [ ] Ensure thread safety review
- [ ] Run `uv run ruff check src/neural_signal/shared/gatekeeper.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 12 — `services/data_loader.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_data_loader.py`
- [ ] Add module docstring
- [ ] Write `test_data_loader_instantiates`
- [ ] Write `test_load_returns_data_bundle`
- [ ] Write `test_data_bundle_has_train_loader`
- [ ] Write `test_data_bundle_has_val_loader`
- [ ] Write `test_data_bundle_has_test_loader`
- [ ] Write `test_train_loader_batch_shape_fcn` — batch x shape (batch_size, 15)
- [ ] Write `test_val_loader_batch_shape_fcn`
- [ ] Write `test_test_loader_batch_shape_fcn`
- [ ] Write `test_train_loader_y_batch_shape` — y shape (batch_size, 1)
- [ ] Write `test_data_bundle_has_raw_X_train`
- [ ] Write `test_data_bundle_has_raw_C_train`
- [ ] Write `test_data_bundle_has_raw_y_train`
- [ ] Write `test_fcn_input_is_concat_of_X_and_C` — shape (N, 15) = (N, 10+5)
- [ ] Write `test_rnn_input_reshape_shape` — shape (N, 10, 1)
- [ ] Write `test_seq_loader_batch_shape_rnn` — batch shape (batch_size, 10, 1)
- [ ] Write `test_data_loader_raises_on_missing_npz`
- [ ] Write `test_data_loader_raises_on_missing_X_train_key`
- [ ] Write `test_data_loader_raises_on_missing_C_train_key`
- [ ] Write `test_data_loader_raises_on_missing_y_train_key`
- [ ] Write `test_train_loader_shuffles_data` — two iterations differ
- [ ] Write `test_val_loader_does_not_shuffle` — two iterations identical
- [ ] Write `test_test_loader_does_not_shuffle`
- [ ] Write `test_batch_size_from_config`
- [ ] Write `test_no_nan_in_train_batch_x`
- [ ] Write `test_no_nan_in_train_batch_y`
- [ ] Write `test_no_inf_in_train_batch_x`
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/services/data_loader.py`
- [ ] Add module docstring
- [ ] Import `numpy`, `torch`, `torch.utils.data`, `pathlib`, `dataclasses`, `constants`
- [ ] Define `DataBundle` dataclass: train_loader, val_loader, test_loader, raw X/C/y per split
- [ ] Add docstring to `DataBundle`
- [ ] Define `DataLoaderService` class with docstring
- [ ] Implement `__init__(self, dataset_path: Path, batch_size: int)`
- [ ] Implement `load() -> DataBundle`
- [ ] Load npz: `np.load(dataset_path)`
- [ ] Validate all 9 keys present (X/C/y × train/val/test)
- [ ] Concatenate `X` and `C` → `x_fcn` shape (N, 15) using `np.concatenate`
- [ ] Reshape `X` → `x_seq` shape (N, 10, 1) for RNN/LSTM
- [ ] Convert to `torch.FloatTensor`
- [ ] Build `TensorDataset(x_fcn, y)` for FCN loaders
- [ ] Build `TensorDataset(x_seq, y)` for sequence loaders
- [ ] Wrap in `DataLoader(shuffle=True)` for train, `shuffle=False` for val/test
- [ ] Return `DataBundle` with all loaders and raw arrays
- [ ] Verify file ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Extract npz key validation into `_validate_keys` method
- [ ] Extract tensor conversion into `_to_tensor` helper
- [ ] Run `uv run ruff check src/neural_signal/services/data_loader.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 13 — `services/preprocessor.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_preprocessor.py`
- [ ] Add module docstring
- [ ] Write `test_preprocessor_instantiates`
- [ ] Write `test_fit_computes_mean_from_train_only`
- [ ] Write `test_fit_computes_std_from_train_only`
- [ ] Write `test_transform_subtracts_mean`
- [ ] Write `test_transform_divides_by_std`
- [ ] Write `test_transform_mean_of_normalized_train_approx_zero`
- [ ] Write `test_transform_std_of_normalized_train_approx_one`
- [ ] Write `test_val_normalized_using_train_mean_not_val_mean`
- [ ] Write `test_test_normalized_using_train_mean_not_test_mean`
- [ ] Write `test_fit_transform_returns_correct_shape`
- [ ] Write `test_no_nan_after_normalization`
- [ ] Write `test_no_inf_after_normalization`
- [ ] Write `test_scaler_params_saved_to_json`
- [ ] Write `test_scaler_params_json_has_mean_key`
- [ ] Write `test_scaler_params_json_has_std_key`
- [ ] Write `test_handles_zero_std_without_division_error`
- [ ] Write `test_raises_before_transform_without_fit`
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/services/preprocessor.py`
- [ ] Add module docstring
- [ ] Import `numpy`, `json`, `pathlib`
- [ ] Define `ScalerParams` dataclass: mean, std
- [ ] Add docstring to `ScalerParams`
- [ ] Define `Preprocessor` class with docstring
- [ ] Implement `__init__(self, scaler_path: Path)`
- [ ] Implement `fit(self, X_train: np.ndarray) -> ScalerParams`
- [ ] Compute mean and std column-wise from train only
- [ ] Handle zero-std columns (replace with 1 to avoid division by zero)
- [ ] Store fitted params in `self._params`
- [ ] Implement `transform(self, X: np.ndarray) -> np.ndarray`
- [ ] Apply `(X - mean) / std` using stored params
- [ ] Raise `RuntimeError` if called before `fit`
- [ ] Implement `fit_transform(self, X_train) -> np.ndarray`
- [ ] Implement `save_params(self)` — write mean and std to JSON at scaler_path
- [ ] Implement `load_params(self)` — load from JSON
- [ ] Verify file ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Ensure `fit` and `transform` are independently callable
- [ ] Run `uv run ruff check src/neural_signal/services/preprocessor.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 14 — `models/fcn.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_fcn.py`
- [ ] Add module docstring
- [ ] Write `test_fcn_instantiates`
- [ ] Write `test_fcn_output_shape_is_batch_by_1` — input (8, 15) → output (8, 1)
- [ ] Write `test_fcn_output_shape_single_sample` — input (1, 15) → output (1, 1)
- [ ] Write `test_fcn_output_shape_large_batch` — input (64, 15) → output (64, 1)
- [ ] Write `test_fcn_forward_no_nan_on_random_input`
- [ ] Write `test_fcn_forward_no_inf_on_random_input`
- [ ] Write `test_fcn_has_three_linear_layers`
- [ ] Write `test_fcn_first_linear_in_features_equals_15`
- [ ] Write `test_fcn_first_linear_out_features_equals_128`
- [ ] Write `test_fcn_second_linear_in_features_equals_128`
- [ ] Write `test_fcn_second_linear_out_features_equals_64`
- [ ] Write `test_fcn_third_linear_in_features_equals_64`
- [ ] Write `test_fcn_third_linear_out_features_equals_1`
- [ ] Write `test_fcn_has_dropout_layers`
- [ ] Write `test_fcn_dropout_rate_matches_config`
- [ ] Write `test_fcn_relu_activation_between_layer1_and_layer2`
- [ ] Write `test_fcn_relu_activation_between_layer2_and_layer3`
- [ ] Write `test_fcn_no_activation_on_output_layer`
- [ ] Write `test_fcn_parameters_require_grad`
- [ ] Write `test_fcn_weight_shapes_correct_layer1` — weight shape (128, 15)
- [ ] Write `test_fcn_weight_shapes_correct_layer2` — weight shape (64, 128)
- [ ] Write `test_fcn_weight_shapes_correct_layer3` — weight shape (1, 64)
- [ ] Write `test_fcn_in_train_mode_dropout_active`
- [ ] Write `test_fcn_in_eval_mode_dropout_inactive`
- [ ] Write `test_fcn_output_deterministic_in_eval_mode`
- [ ] Write `test_fcn_built_from_config_hidden_sizes`
- [ ] Write `test_fcn_built_from_config_dropout_rate`
- [ ] Write `test_fcn_hidden_sizes_configurable_not_hardcoded`
- [ ] Write `test_fcn_forward_output_is_float_tensor`
- [ ] Write `test_fcn_forward_with_zero_input_no_error`
- [ ] Write `test_fcn_forward_with_negative_input_no_error`
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/models/fcn.py`
- [ ] Add module docstring
- [ ] Import `torch`, `torch.nn`, `dataclasses`
- [ ] Import `FCNConfig` from `shared.config`
- [ ] Define `FCNModel(nn.Module)` class with docstring
- [ ] Implement `__init__(self, cfg: FCNConfig)`
- [ ] Build `self.network = nn.Sequential(...)` using `cfg.hidden_sizes` and `cfg.dropout_rate`
- [ ] Layer 1: `nn.Linear(input_size, cfg.hidden_sizes[0])` → `nn.ReLU()` → `nn.Dropout(cfg.dropout_rate)`
- [ ] Layer 2: `nn.Linear(cfg.hidden_sizes[0], cfg.hidden_sizes[1])` → `nn.ReLU()` → `nn.Dropout(cfg.dropout_rate)`
- [ ] Output: `nn.Linear(cfg.hidden_sizes[1], cfg.output_size)`
- [ ] No activation on output layer (linear)
- [ ] Read `input_size` from constants (`INPUT_SIZE_FCN = 15`)
- [ ] Implement `forward(self, x: torch.Tensor) -> torch.Tensor`
- [ ] `return self.network(x)`
- [ ] Verify no hardcoded 128, 64, 0.1 in model code — all from cfg
- [ ] Verify file ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Confirm `nn.Sequential` is used cleanly — no redundant wrappers
- [ ] Confirm docstrings on `__init__` and `forward`
- [ ] Run `uv run ruff check src/neural_signal/models/fcn.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 15 — `models/rnn.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_rnn.py`
- [ ] Add module docstring
- [ ] Write `test_rnn_instantiates`
- [ ] Write `test_rnn_output_shape_is_batch_by_1` — input (8, 10, 1) → output (8, 1)
- [ ] Write `test_rnn_output_shape_single_sample` — input (1, 10, 1) → output (1, 1)
- [ ] Write `test_rnn_output_shape_large_batch` — input (64, 10, 1) → output (64, 1)
- [ ] Write `test_rnn_forward_no_nan_on_random_input`
- [ ] Write `test_rnn_forward_no_inf_on_random_input`
- [ ] Write `test_rnn_has_rnn_layer`
- [ ] Write `test_rnn_hidden_size_equals_64`
- [ ] Write `test_rnn_nonlinearity_equals_tanh`
- [ ] Write `test_rnn_num_layers_equals_1`
- [ ] Write `test_rnn_batch_first_equals_true`
- [ ] Write `test_rnn_has_linear_head`
- [ ] Write `test_rnn_linear_head_in_features_equals_64`
- [ ] Write `test_rnn_linear_head_out_features_equals_1`
- [ ] Write `test_rnn_many_to_one_last_hidden_state_used` — mock RNN output, verify index -1 extracted
- [ ] Write `test_rnn_parameters_require_grad`
- [ ] Write `test_rnn_hidden_size_from_config`
- [ ] Write `test_rnn_nonlinearity_from_config`
- [ ] Write `test_rnn_output_is_float_tensor`
- [ ] Write `test_rnn_forward_with_zero_input_no_error`
- [ ] Write `test_rnn_input_size_equals_1` — single feature per time step
- [ ] Write `test_rnn_output_not_equal_for_different_sequences` — model is not constant
- [ ] Write `test_rnn_hidden_size_not_hardcoded` — change hidden_size in config, verify model changes
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/models/rnn.py`
- [ ] Add module docstring
- [ ] Import `torch`, `torch.nn`
- [ ] Import `RNNConfig` from `shared.config`
- [ ] Define `RNNModel(nn.Module)` class with docstring
- [ ] Implement `__init__(self, cfg: RNNConfig)`
- [ ] Instantiate `self.rnn = nn.RNN(input_size=1, hidden_size=cfg.hidden_size, nonlinearity=cfg.nonlinearity, batch_first=True, num_layers=cfg.num_layers)`
- [ ] Instantiate `self.fc = nn.Linear(cfg.hidden_size, cfg.output_size)`
- [ ] Implement `forward(self, x: torch.Tensor) -> torch.Tensor`
- [ ] Run `out, _ = self.rnn(x)` — out shape (batch, seq_len, hidden_size)
- [ ] Extract last time step: `last = out[:, -1, :]` — shape (batch, hidden_size)
- [ ] Apply `self.fc(last)` → output shape (batch, 1)
- [ ] Return output
- [ ] Verify no hardcoded 64 in model code — all from cfg
- [ ] Verify file ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Confirm `forward` docstring explains many-to-one extraction
- [ ] Run `uv run ruff check src/neural_signal/models/rnn.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 16 — `models/lstm.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_lstm.py`
- [ ] Add module docstring
- [ ] Write `test_lstm_instantiates`
- [ ] Write `test_lstm_output_shape_is_batch_by_1` — input (8, 10, 1) → output (8, 1)
- [ ] Write `test_lstm_output_shape_single_sample` — input (1, 10, 1) → output (1, 1)
- [ ] Write `test_lstm_output_shape_large_batch` — input (64, 10, 1) → output (64, 1)
- [ ] Write `test_lstm_forward_no_nan_on_random_input`
- [ ] Write `test_lstm_forward_no_inf_on_random_input`
- [ ] Write `test_lstm_has_lstm_layer`
- [ ] Write `test_lstm_hidden_size_equals_64`
- [ ] Write `test_lstm_num_layers_equals_1`
- [ ] Write `test_lstm_batch_first_equals_true`
- [ ] Write `test_lstm_has_two_dense_layers_in_head`
- [ ] Write `test_lstm_first_dense_in_features_equals_64`
- [ ] Write `test_lstm_first_dense_out_features_equals_32`
- [ ] Write `test_lstm_second_dense_in_features_equals_32`
- [ ] Write `test_lstm_second_dense_out_features_equals_1`
- [ ] Write `test_lstm_relu_between_dense_layers`
- [ ] Write `test_lstm_no_activation_on_output`
- [ ] Write `test_lstm_many_to_one_last_output_used` — verify index -1 extracted from LSTM output
- [ ] Write `test_lstm_parameters_require_grad`
- [ ] Write `test_lstm_hidden_size_from_config`
- [ ] Write `test_lstm_dense_hidden_size_from_config`
- [ ] Write `test_lstm_output_is_float_tensor`
- [ ] Write `test_lstm_forward_with_zero_input_no_error`
- [ ] Write `test_lstm_input_size_equals_1`
- [ ] Write `test_lstm_output_not_equal_for_different_sequences`
- [ ] Write `test_lstm_hidden_size_not_hardcoded`
- [ ] Write `test_lstm_dense_hidden_size_not_hardcoded`
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/models/lstm.py`
- [ ] Add module docstring
- [ ] Import `torch`, `torch.nn`
- [ ] Import `LSTMConfig` from `shared.config`
- [ ] Define `LSTMModel(nn.Module)` class with docstring
- [ ] Implement `__init__(self, cfg: LSTMConfig)`
- [ ] Instantiate `self.lstm = nn.LSTM(input_size=1, hidden_size=cfg.hidden_size, batch_first=True, num_layers=cfg.num_layers)`
- [ ] Instantiate `self.fc1 = nn.Linear(cfg.hidden_size, cfg.dense_hidden_size)`
- [ ] Instantiate `self.relu = nn.ReLU()`
- [ ] Instantiate `self.fc2 = nn.Linear(cfg.dense_hidden_size, cfg.output_size)`
- [ ] Implement `forward(self, x: torch.Tensor) -> torch.Tensor`
- [ ] Run `out, _ = self.lstm(x)` — out shape (batch, seq_len, hidden_size)
- [ ] Extract last time step: `last = out[:, -1, :]`
- [ ] Apply `self.fc1(last)` → ReLU → `self.fc2` → output (batch, 1)
- [ ] Return output
- [ ] Verify no hardcoded 64, 32 in model code — all from cfg
- [ ] Verify file ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Confirm `forward` docstring explains dense head structure
- [ ] Run `uv run ruff check src/neural_signal/models/lstm.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 17 — `services/trainer.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_trainer.py`
- [ ] Add module docstring
- [ ] Write `test_trainer_instantiates`
- [ ] Write `test_trainer_returns_training_result`
- [ ] Write `test_training_result_has_best_val_mse`
- [ ] Write `test_training_result_has_epochs_trained`
- [ ] Write `test_training_result_has_stopped_early_flag`
- [ ] Write `test_training_result_has_checkpoint_path`
- [ ] Write `test_train_loss_is_finite` — MSE is not NaN or Inf after 1 epoch
- [ ] Write `test_val_loss_is_finite`
- [ ] Write `test_train_loss_decreases_over_epochs` — loss[5] < loss[0] on simple data
- [ ] Write `test_optimizer_is_adam` — check optimizer class
- [ ] Write `test_learning_rate_from_config`
- [ ] Write `test_weight_decay_applied_for_fcn` — check optimizer.param_groups weight_decay
- [ ] Write `test_loss_function_is_mse`
- [ ] Write `test_checkpoint_saved_to_configured_path`
- [ ] Write `test_checkpoint_loadable_by_torch`
- [ ] Write `test_early_stopping_stops_before_max_epochs`
- [ ] Write `test_early_stopping_patience_from_config`
- [ ] Write `test_early_stopping_does_not_trigger_if_val_improving`
- [ ] Write `test_stopped_early_flag_true_when_early_stop_fires`
- [ ] Write `test_stopped_early_flag_false_when_training_completes`
- [ ] Write `test_epochs_trained_equals_epoch_count_not_max_epochs`
- [ ] Write `test_training_log_csv_created`
- [ ] Write `test_training_log_has_epoch_column`
- [ ] Write `test_training_log_has_train_mse_column`
- [ ] Write `test_training_log_has_val_mse_column`
- [ ] Write `test_training_log_row_count_equals_epochs_trained`
- [ ] Write `test_best_checkpoint_is_minimum_val_mse_epoch` — not last epoch
- [ ] Write `test_random_seed_fixes_weight_init`
- [ ] Write `test_batch_size_from_config`
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/services/trainer.py`
- [ ] Add module docstring
- [ ] Import `torch`, `torch.nn`, `torch.optim`, `pathlib`, `csv`, `dataclasses`, `typing`
- [ ] Import `TrainingConfig` from `shared.config`
- [ ] Define `TrainingResult` dataclass: best_val_mse, epochs_trained, stopped_early, checkpoint_path, log_path
- [ ] Add docstring to `TrainingResult`
- [ ] Define `Trainer` class with docstring
- [ ] Implement `__init__(self, cfg: TrainingConfig, checkpoint_path: Path, log_path: Path)`
- [ ] Implement `train(self, model: nn.Module, train_loader, val_loader) -> TrainingResult`
- [ ] Set random seed: `torch.manual_seed(cfg.random_seed)`
- [ ] Instantiate optimizer: `torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)`
- [ ] Instantiate loss: `nn.MSELoss()`
- [ ] Initialize best_val_mse = infinity, patience_counter = 0
- [ ] Epoch loop (range cfg.max_epochs):
  - [ ] Train loop: `model.train()`, iterate batches, forward, loss, backward, optimizer step
  - [ ] Val loop: `model.eval()`, `torch.no_grad()`, compute val MSE
  - [ ] Update best checkpoint if val_mse improved
  - [ ] Increment patience_counter or reset
  - [ ] Break if patience_counter > patience
  - [ ] Append row to log (epoch, train_mse, val_mse)
- [ ] Save checkpoint: `torch.save(model.state_dict(), checkpoint_path)`
- [ ] Write CSV log using `csv.writer`
- [ ] Return `TrainingResult`
- [ ] Implement private `_run_epoch(model, loader, optimizer, loss_fn, training) -> float`
- [ ] Implement private `_save_checkpoint(model, path)`
- [ ] Verify file ≤ 150 lines (split if needed)
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Confirm early stopping logic in its own private method
- [ ] Confirm train/val loops cleanly separated
- [ ] Run `uv run ruff check src/neural_signal/services/trainer.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 18 — `services/evaluator.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_evaluator.py`
- [ ] Add module docstring
- [ ] Write `test_evaluator_instantiates`
- [ ] Write `test_evaluate_returns_eval_result`
- [ ] Write `test_eval_result_has_train_mse`
- [ ] Write `test_eval_result_has_val_mse`
- [ ] Write `test_eval_result_has_test_mse`
- [ ] Write `test_mse_is_finite`
- [ ] Write `test_mse_is_non_negative`
- [ ] Write `test_mse_equals_zero_for_perfect_prediction`
- [ ] Write `test_mse_computed_without_gradient`
- [ ] Write `test_evaluate_all_returns_dataframe`
- [ ] Write `test_comparison_dataframe_has_model_column`
- [ ] Write `test_comparison_dataframe_has_train_mse_column`
- [ ] Write `test_comparison_dataframe_has_val_mse_column`
- [ ] Write `test_comparison_dataframe_has_test_mse_column`
- [ ] Write `test_comparison_dataframe_has_epochs_trained_column`
- [ ] Write `test_comparison_dataframe_has_stopped_early_column`
- [ ] Write `test_comparison_dataframe_has_three_rows` — one per model
- [ ] Write `test_comparison_dataframe_saved_to_csv`
- [ ] Write `test_comparison_csv_loadable_by_pandas`
- [ ] Write `test_checkpoint_loaded_before_eval` — verify state dict loaded from file
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/services/evaluator.py`
- [ ] Add module docstring
- [ ] Import `torch`, `torch.nn`, `pandas`, `pathlib`, `dataclasses`
- [ ] Import `constants` for column names
- [ ] Define `EvalResult` dataclass: model_name, train_mse, val_mse, test_mse, epochs_trained, stopped_early
- [ ] Add docstring to `EvalResult`
- [ ] Define `Evaluator` class with docstring
- [ ] Implement `__init__(self, loss_fn: nn.Module, table_path: Path)`
- [ ] Implement `evaluate(self, model, loader) -> float` — compute MSE with `torch.no_grad()`
- [ ] Implement `evaluate_all(self, models_dict, loaders_dict, results_list) -> pd.DataFrame`
- [ ] Build DataFrame from `EvalResult` list using `RESULT_COL_*` constants
- [ ] Save to CSV at `table_path`
- [ ] Return DataFrame
- [ ] Implement `load_checkpoint(self, model, checkpoint_path) -> nn.Module`
- [ ] Verify file ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Confirm `evaluate` uses `model.eval()` before inference
- [ ] Run `uv run ruff check src/neural_signal/services/evaluator.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 19 — `services/visualizer.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_visualizer.py`
- [ ] Add module docstring
- [ ] Write `test_visualizer_instantiates`
- [ ] Write `test_plot_clean_noisy_predicted_creates_file`
- [ ] Write `test_plot_clean_noisy_predicted_file_is_png`
- [ ] Write `test_plot_clean_noisy_predicted_file_not_empty`
- [ ] Write `test_plot_clean_noisy_predicted_has_five_lines` — clean, noisy, fcn, rnn, lstm
- [ ] Write `test_plot_loss_curves_creates_file_for_fcn`
- [ ] Write `test_plot_loss_curves_creates_file_for_rnn`
- [ ] Write `test_plot_loss_curves_creates_file_for_lstm`
- [ ] Write `test_plot_mse_comparison_creates_file`
- [ ] Write `test_plot_mse_comparison_file_not_empty`
- [ ] Write `test_plot_residuals_creates_file_for_each_model`
- [ ] Write `test_plot_pred_vs_actual_creates_file`
- [ ] Write `test_plot_uses_correct_colors_from_constants`
- [ ] Write `test_plot_raises_on_mismatched_array_lengths`
- [ ] Write `test_plot_works_with_minimal_data` — 10 samples, no error
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/services/visualizer.py`
- [ ] Add module docstring
- [ ] Import `matplotlib.pyplot`, `seaborn`, `numpy`, `pathlib`, `pandas`
- [ ] Import `constants` for color names
- [ ] Define `Visualizer` class with docstring
- [ ] Implement `__init__(self, results_dir: Path)`
- [ ] Implement `plot_clean_noisy_predicted(self, clean, noisy, fcn_pred, rnn_pred, lstm_pred, t)`
  - [ ] Create single figure with 5 overlay lines
  - [ ] Clean: green solid; Noisy: grey dashed; FCN: red; RNN: orange; LSTM: purple
  - [ ] Add legend, x-label "Time (s)", y-label "Amplitude", title
  - [ ] Save to `results_dir / "clean_noisy_predicted.png"` at 300 DPI
  - [ ] Close figure
- [ ] Implement `plot_loss_curves(self, model_name, train_losses, val_losses)`
  - [ ] Line plot: train (blue) vs val (orange) per epoch
  - [ ] Add legend, labels, title "Training Loss — {model_name}"
  - [ ] Save to `results_dir / f"loss_curves_{model_name}.png"` at 300 DPI
  - [ ] Close figure
- [ ] Implement `plot_mse_comparison(self, comparison_df: pd.DataFrame)`
  - [ ] Grouped bar chart: train/val/test MSE per model
  - [ ] Add value labels above bars
  - [ ] Save to `results_dir / "mse_comparison.png"` at 300 DPI
  - [ ] Close figure
- [ ] Implement `plot_residuals(self, model_name, y_true, y_pred)`
  - [ ] Scatter of residuals (y_pred - y_true) vs sample index
  - [ ] Horizontal dashed line at 0
  - [ ] Save to `results_dir / f"residuals_{model_name}.png"` at 300 DPI
  - [ ] Close figure
- [ ] Implement `plot_pred_vs_actual(self, y_true, predictions_dict)`
  - [ ] Three scatter panels (FCN, RNN, LSTM): predicted vs actual
  - [ ] Diagonal identity line for reference
  - [ ] Save to `results_dir / "pred_vs_actual.png"` at 300 DPI
  - [ ] Close figure
- [ ] Verify file ≤ 150 lines (split into `visualizer_plots.py` mixin if needed)
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Extract `_save_and_close(fig, path)` helper
- [ ] Extract `_apply_common_style(ax)` helper for labels/grid
- [ ] Confirm all figures closed after saving (no memory leak)
- [ ] Run `uv run ruff check src/neural_signal/services/visualizer.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 20 — `sdk/sdk.py` — TDD

### RED Phase
- [ ] Create `tests/unit/test_sdk.py`
- [ ] Add module docstring
- [ ] Write `test_sdk_instantiates_with_config_path`
- [ ] Write `test_sdk_run_all_returns_run_result`
- [ ] Write `test_run_result_has_fcn_eval`
- [ ] Write `test_run_result_has_rnn_eval`
- [ ] Write `test_run_result_has_lstm_eval`
- [ ] Write `test_run_result_has_comparison_df`
- [ ] Write `test_sdk_train_model_fcn_returns_training_result` (mock services)
- [ ] Write `test_sdk_train_model_rnn_returns_training_result` (mock services)
- [ ] Write `test_sdk_train_model_lstm_returns_training_result` (mock services)
- [ ] Write `test_sdk_raises_on_unknown_model_name`
- [ ] Write `test_sdk_evaluate_all_returns_dataframe`
- [ ] Write `test_sdk_get_version_returns_string`
- [ ] Write `test_sdk_calls_data_loader_service` (mock)
- [ ] Write `test_sdk_calls_preprocessor` (mock)
- [ ] Write `test_sdk_calls_trainer_for_each_model` (mock)
- [ ] Write `test_sdk_calls_evaluator` (mock)
- [ ] Write `test_sdk_calls_visualizer` (mock)
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

### GREEN Phase
- [ ] Create `src/neural_signal/sdk/sdk.py`
- [ ] Add module docstring — explain this is the sole public API
- [ ] Import all service and model classes
- [ ] Import `version`, `constants`
- [ ] Define `RunResult` dataclass: fcn_eval, rnn_eval, lstm_eval, comparison_df
- [ ] Add docstring to `RunResult`
- [ ] Define `NeuralSignalSDK` class with docstring
- [ ] Implement `__init__(self, config_path, rate_limits_path)`
  - [ ] Instantiate `ConfigManager` and call `load()`
  - [ ] Instantiate `ApiGatekeeper`
  - [ ] Store `self._cfg`
- [ ] Implement `run_all(self) -> RunResult`
  - [ ] Step 1: `DataLoaderService.load()` → DataBundle
  - [ ] Step 2: `Preprocessor.fit_transform(X_train)`, transform val/test
  - [ ] Step 3: rebuild DataLoaders with normalized X
  - [ ] Step 4: for each model in ["fcn", "rnn", "lstm"]: call `train_model(name, loaders)`
  - [ ] Step 5: `Evaluator.evaluate_all()` → comparison_df
  - [ ] Step 6: `Visualizer.plot_*()` — all mandatory plots
  - [ ] Return `RunResult`
- [ ] Implement `train_model(self, model_name: str, ...) -> TrainingResult`
  - [ ] Build model from config (FCNModel / RNNModel / LSTMModel)
  - [ ] Instantiate Trainer with correct checkpoint path
  - [ ] Call `trainer.train(model, train_loader, val_loader)`
  - [ ] Return TrainingResult
- [ ] Implement `evaluate_all(self) -> pd.DataFrame`
- [ ] Implement `get_version(self) -> str`
- [ ] Raise `ValueError` on unknown model_name
- [ ] Verify file ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

### REFACTOR Phase
- [ ] Confirm no business logic in SDK — only orchestration
- [ ] Confirm each service instantiated exactly once
- [ ] Run `uv run ruff check src/neural_signal/sdk/sdk.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 21 — `main.py` CLI Entry Point

- [ ] Create `src/neural_signal/main.py`
- [ ] Add module docstring
- [ ] Import `argparse`, `pathlib`, `sys`, `NeuralSignalSDK`
- [ ] Define `parse_args() -> argparse.Namespace`
- [ ] Add `--config` argument (default: `config/setup.json`)
- [ ] Add `--rate-limits` argument (default: `config/rate_limits.json`)
- [ ] Add `--model` argument (choices: fcn, rnn, lstm, all; default: all)
- [ ] Add `--verbose` flag
- [ ] Define `main() -> int`
- [ ] Parse args, instantiate `NeuralSignalSDK`
- [ ] If `args.model == "all"`: call `sdk.run_all()`
- [ ] Else: call `sdk.train_model(args.model)` then `sdk.evaluate_all()`
- [ ] Print summary: model results, paths saved, time elapsed
- [ ] Return exit code 0 on success, 1 on error
- [ ] Add `if __name__ == "__main__": sys.exit(main())`
- [ ] Test: `uv run python -m neural_signal --help` outputs usage
- [ ] Test: `uv run python -m neural_signal --model fcn` runs without error
- [ ] Verify file ≤ 150 lines
- [ ] Verify `uv run ruff check src/neural_signal/main.py` → 0 errors

---

## Phase 22 — Per-Algorithm PRDs

### `docs/PRD_fcn.md`
- [ ] Create `docs/PRD_fcn.md`
- [ ] Add title and version header
- [ ] Document FCN input data: shape (batch, 15), concatenated X + C
- [ ] Document FCN output data: shape (batch, 1), scalar regression value
- [ ] Document FCN setup data: hidden_sizes, dropout_rate, weight_decay, all from config
- [ ] Write layer-by-layer forward pass pseudocode
- [ ] Document Dropout behavior: active during training, inactive during eval
- [ ] Document L2 regularization mechanism (weight_decay in Adam)
- [ ] Document expected MSE range based on signal noise levels
- [ ] List acceptance criteria for FCN
- [ ] Add OAT plan: vary dropout_rate, hidden_sizes, weight_decay

### `docs/PRD_rnn.md`
- [ ] Create `docs/PRD_rnn.md`
- [ ] Add title and version header
- [ ] Document RNN input data: shape (batch, seq_len=10, 1)
- [ ] Document C injection strategy: prepend or concatenate to sequence
- [ ] Document RNN output data: shape (batch, 1)
- [ ] Document many-to-one architecture: last hidden state `h_t[:, -1, :]`
- [ ] Document tanh non-linearity choice vs ReLU
- [ ] Document gradient vanishing risk and mitigation (gradient clipping option)
- [ ] Write forward pass pseudocode
- [ ] List acceptance criteria for RNN
- [ ] Add OAT plan: vary hidden_size, num_layers, seq_len

### `docs/PRD_lstm.md`
- [ ] Create `docs/PRD_lstm.md`
- [ ] Add title and version header
- [ ] Document LSTM input data: shape (batch, seq_len=10, 1)
- [ ] Document LSTM gates: forget, input, cell, output
- [ ] Document LSTM output data: shape (batch, 1)
- [ ] Document many-to-one architecture: last output `out[:, -1, :]`
- [ ] Document dense head: Linear(64,32) → ReLU → Linear(32,1)
- [ ] Write forward pass pseudocode
- [ ] List acceptance criteria for LSTM
- [ ] Add OAT plan: vary hidden_size, dense_hidden_size, num_layers

---

## Phase 23 — Integration Tests

- [ ] Create `tests/integration/test_pipeline.py`
- [ ] Add module docstring
- [ ] Write `test_full_pipeline_runs_without_error` — NeuralSignalSDK.run_all() end-to-end
- [ ] Write `test_run_all_creates_comparison_table_csv`
- [ ] Write `test_run_all_creates_clean_noisy_predicted_png`
- [ ] Write `test_run_all_creates_loss_curves_fcn_png`
- [ ] Write `test_run_all_creates_loss_curves_rnn_png`
- [ ] Write `test_run_all_creates_loss_curves_lstm_png`
- [ ] Write `test_run_all_creates_mse_comparison_png`
- [ ] Write `test_run_all_creates_fcn_checkpoint`
- [ ] Write `test_run_all_creates_rnn_checkpoint`
- [ ] Write `test_run_all_creates_lstm_checkpoint`
- [ ] Write `test_comparison_table_has_three_rows`
- [ ] Write `test_all_mse_values_are_finite`
- [ ] Write `test_all_mse_values_are_non_negative`
- [ ] Write `test_fcn_train_mse_less_than_random_baseline`
- [ ] Write `test_rnn_train_mse_less_than_random_baseline`
- [ ] Write `test_lstm_train_mse_less_than_random_baseline`
- [ ] Write `test_early_stopping_log_entries_match_epochs_trained`
- [ ] Write `test_checkpoint_loadable_and_produces_same_predictions`
- [ ] Write `test_fcn_train_mse_less_than_val_mse_or_equal` — no severe underfitting
- [ ] Write `test_pipeline_reproducible_with_same_seed`
- [ ] Verify test file ≤ 150 lines
- [ ] Run integration tests — confirm all PASS

---

## Phase 24 — Quality Gates

- [ ] Run `uv run ruff check src/` → 0 errors
- [ ] Run `uv run ruff check tests/` → 0 errors
- [ ] Fix any ruff issues found in `src/neural_signal/__init__.py`
- [ ] Fix any ruff issues in `src/neural_signal/constants.py`
- [ ] Fix any ruff issues in `src/neural_signal/main.py`
- [ ] Fix any ruff issues in `src/neural_signal/sdk/sdk.py`
- [ ] Fix any ruff issues in `src/neural_signal/models/fcn.py`
- [ ] Fix any ruff issues in `src/neural_signal/models/rnn.py`
- [ ] Fix any ruff issues in `src/neural_signal/models/lstm.py`
- [ ] Fix any ruff issues in `src/neural_signal/services/data_loader.py`
- [ ] Fix any ruff issues in `src/neural_signal/services/preprocessor.py`
- [ ] Fix any ruff issues in `src/neural_signal/services/trainer.py`
- [ ] Fix any ruff issues in `src/neural_signal/services/evaluator.py`
- [ ] Fix any ruff issues in `src/neural_signal/services/visualizer.py`
- [ ] Fix any ruff issues in `src/neural_signal/shared/config.py`
- [ ] Fix any ruff issues in `src/neural_signal/shared/gatekeeper.py`
- [ ] Fix any ruff issues in `src/neural_signal/shared/version.py`
- [ ] Run `uv run pytest tests/unit/` → all pass
- [ ] Run `uv run pytest tests/integration/` → all pass
- [ ] Run `uv run pytest tests/ --cov=src --cov-report=term-missing` → ≥ 85%
- [ ] Verify `src/neural_signal/__init__.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/constants.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/main.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/sdk/sdk.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/models/fcn.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/models/rnn.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/models/lstm.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/services/data_loader.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/services/preprocessor.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/services/trainer.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/services/evaluator.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/services/visualizer.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/shared/config.py` ≤ 150 lines
- [ ] Verify `src/neural_signal/shared/gatekeeper.py` ≤ 150 lines
- [ ] Verify `tests/conftest.py` ≤ 150 lines
- [ ] Verify all test files ≤ 150 lines
- [ ] Grep for hardcoded `128` in model files — must not appear outside config
- [ ] Grep for hardcoded `64` in model files — must not appear outside config
- [ ] Grep for hardcoded `0.001` in Python source — must come from config
- [ ] Grep for hardcoded `0.1` in model files — dropout must come from config
- [ ] Grep for hardcoded `32` in model files — dense hidden size must come from config
- [ ] Verify `.env` not in git index
- [ ] Verify `uv.lock` committed and up to date

---

## Phase 25 — OAT Parameter Sensitivity Analysis

- [ ] Plan OAT sweep parameters: window_size, hidden_size, learning_rate, dropout_rate, batch_size
- [ ] Define parameter ranges for window_size: [5, 10, 20, 30, 50]
- [ ] Define parameter ranges for hidden_size: [16, 32, 64, 128, 256]
- [ ] Define parameter ranges for learning_rate: [0.0001, 0.0005, 0.001, 0.005, 0.01]
- [ ] Define parameter ranges for dropout_rate: [0.0, 0.1, 0.2, 0.3, 0.5]
- [ ] Define parameter ranges for batch_size: [16, 32, 64, 128, 256]
- [ ] Implement sweep loop in notebook or `services/sensitivity.py`
- [ ] For each parameter × value: train all 3 models, record val MSE
- [ ] Save sensitivity results to `results/sensitivity/sensitivity_results.csv`
- [ ] Plot heatmap: model × parameter value → val MSE
- [ ] Plot line chart per parameter: val MSE vs parameter value, 3 lines (FCN/RNN/LSTM)
- [ ] Save heatmap to `results/sensitivity/sensitivity_heatmap.png`
- [ ] Save line charts to `results/sensitivity/sensitivity_<param>.png`
- [ ] Write summary: which parameters most affect each model
- [ ] Add sensitivity findings to notebook

---

## Phase 26 — Jupyter Notebook: `notebooks/results_analysis.ipynb`

- [ ] Create `notebooks/results_analysis.ipynb`
- [ ] Cell 1 — Markdown title: "# Neural Network Signal Regression — Results Analysis"
- [ ] Cell 2 — Imports: torch, numpy, matplotlib, seaborn, pandas, pathlib, sys
- [ ] Cell 3 — Set sys.path to include src/
- [ ] Cell 4 — Import NeuralSignalSDK, instantiate with config paths
- [ ] Cell 5 — Markdown: "## 1. Dataset Overview"
- [ ] Cell 6 — Load dataset.npz, print shapes of all 9 arrays
- [ ] Cell 7 — Print split sizes and percentages
- [ ] Cell 8 — Markdown: "## 2. Model Architectures"
- [ ] Cell 9 — Instantiate FCNModel, print architecture (`print(model)`)
- [ ] Cell 10 — Count FCN parameters: `sum(p.numel() for p in model.parameters())`
- [ ] Cell 11 — Instantiate RNNModel, print architecture
- [ ] Cell 12 — Count RNN parameters
- [ ] Cell 13 — Instantiate LSTMModel, print architecture
- [ ] Cell 14 — Count LSTM parameters
- [ ] Cell 15 — Markdown: "## 3. Training Results"
- [ ] Cell 16 — Load comparison_table.csv, display as DataFrame
- [ ] Cell 17 — Bar chart: train/val/test MSE per model
- [ ] Cell 18 — Markdown table: model | train_mse | val_mse | test_mse | epochs | early_stop
- [ ] Cell 19 — Markdown: "## 4. Training Loss Curves"
- [ ] Cell 20 — Load training_log_fcn.csv, plot train vs val loss per epoch
- [ ] Cell 21 — Load training_log_rnn.csv, plot train vs val loss per epoch
- [ ] Cell 22 — Load training_log_lstm.csv, plot train vs val loss per epoch
- [ ] Cell 23 — Markdown: "## 5. Signal Reconstruction"
- [ ] Cell 24 — Load best checkpoints for all 3 models
- [ ] Cell 25 — Run inference on test set (first 500 samples)
- [ ] Cell 26 — Plot: Clean vs. Noisy vs. FCN vs. RNN vs. LSTM (same figure)
- [ ] Cell 27 — Markdown: "## 6. Residual Analysis"
- [ ] Cell 28 — Plot residuals for FCN on test set
- [ ] Cell 29 — Plot residuals for RNN on test set
- [ ] Cell 30 — Plot residuals for LSTM on test set
- [ ] Cell 31 — Print: mean, std, max absolute residual per model
- [ ] Cell 32 — Markdown: "## 7. Parameter Sensitivity (OAT)"
- [ ] Cell 33 — Load sensitivity_results.csv
- [ ] Cell 34 — Plot sensitivity line charts per parameter
- [ ] Cell 35 — Heatmap: model × parameter value → val MSE
- [ ] Cell 36 — Markdown: "## 8. Statistical Comparison"
- [ ] Cell 37 — Paired t-test: FCN vs RNN test MSE over multiple runs
- [ ] Cell 38 — Paired t-test: FCN vs LSTM test MSE
- [ ] Cell 39 — Paired t-test: RNN vs LSTM test MSE
- [ ] Cell 40 — Summary table of statistical significance
- [ ] Cell 41 — Markdown: "## 9. Conclusions"
- [ ] Cell 42 — Which model has lowest test MSE?
- [ ] Cell 43 — Which model converged fastest (fewest epochs)?
- [ ] Cell 44 — Which model is most sensitive to hyperparameters?
- [ ] Cell 45 — Final recommendation and justification
- [ ] Run all cells — confirm no errors
- [ ] Save notebook with outputs: `jupyter nbconvert --to notebook --execute`

---

## Phase 27 — README.md

- [ ] Open README.md at project root
- [ ] Update project title to "Neural Network Signal Regression — FCN · RNN · LSTM"
- [ ] Add overview paragraph: purpose, dataset, three architectures
- [ ] Add "## Model Architectures" section with table (model, layers, params)
- [ ] Add "## Training Configuration" section: Adam, MSE, batch=64, early stopping
- [ ] Add "## Quickstart" section
- [ ] Add step: `uv sync`
- [ ] Add step: `uv run python -m neural_signal` — train all models
- [ ] Add step: `uv run python -m neural_signal --model fcn` — single model
- [ ] Add "## Project Structure" section with annotated directory tree
- [ ] Add "## Configuration" section explaining `config/setup.json` fields
- [ ] Add "## Results" section with embedded visualization images
- [ ] Embed `results/clean_noisy_predicted.png`
- [ ] Embed `results/mse_comparison.png`
- [ ] Add "## Testing" section: `uv run pytest tests/`, coverage
- [ ] Add "## Linting" section: `uv run ruff check`
- [ ] Add "## Phase 1 Reference" link to dataset generation docs
- [ ] Add "## License" section
- [ ] Add "## Author" section with contact email
- [ ] Verify README renders correctly as markdown

---

## Phase 28 — Prompt Engineering Log

- [ ] Open `docs/prompt_engineering_log.md`
- [ ] Document Prompt 1: FCN/RNN/LSTM architecture description given to AI
- [ ] Document Prompt 2: PRD/PLAN/TODO generation request
- [ ] Document AI model used: claude-sonnet-4-6
- [ ] Document date of each prompt: 2026-05-10
- [ ] Document outcome of each prompt
- [ ] Document any corrections applied to AI output
- [ ] Document Phase 10-20 implementation prompts as they occur
- [ ] Keep log updated throughout all phases

---

## Phase 29 — Final Submission Checklist

- [ ] All docs present: PRD.md, PLAN.md, TODO.md, PRD_fcn.md, PRD_rnn.md, PRD_lstm.md, README.md
- [ ] Prompt engineering log complete and up-to-date
- [ ] `uv run ruff check` → 0 errors across entire project
- [ ] `uv run pytest tests/ --cov=src` → ≥ 85% coverage
- [ ] All source files ≤ 150 lines — verified per file
- [ ] All test files ≤ 150 lines — verified per file
- [ ] No hardcoded hyperparameters in Python source
- [ ] No API keys or secrets in source
- [ ] `.env-example` committed with dummy values
- [ ] `.env` not tracked by git
- [ ] `uv.lock` committed and up to date
- [ ] `pyproject.toml` is sole dependency declaration — no requirements.txt
- [ ] No `pip install` used anywhere in project
- [ ] `results/clean_noisy_predicted.png` exists and shows all 5 overlaid signals
- [ ] `results/comparison_table.csv` exists with 3 rows and 6 columns
- [ ] `results/loss_curves_fcn.png`, `_rnn.png`, `_lstm.png` all exist
- [ ] `results/mse_comparison.png` exists
- [ ] `results/residuals_fcn.png`, `_rnn.png`, `_lstm.png` all exist
- [ ] `results/pred_vs_actual.png` exists
- [ ] `results/checkpoints/fcn_best.pt` exists and loadable
- [ ] `results/checkpoints/rnn_best.pt` exists and loadable
- [ ] `results/checkpoints/lstm_best.pt` exists and loadable
- [ ] `results/sensitivity/` directory populated with sweep results
- [ ] `notebooks/results_analysis.ipynb` executed end-to-end without errors
- [ ] All notebook cells have output saved
- [ ] `NeuralSignalSDK` is sole entry point — no service imported directly in main.py
- [ ] `ApiGatekeeper` wired for all rate-limited calls
- [ ] `ConfigManager` used for all config reads — no direct JSON parsing outside shared/config.py
- [ ] `version.py` reflects correct version string "1.00"
- [ ] Git history has structured commit messages (one commit per logical unit)
- [ ] All three models show val MSE < random baseline MSE
- [ ] Early stopping log shows patience correctly applied
- [ ] Statistical comparison (t-tests) documented in notebook
- [ ] Supervisor final review completed
- [ ] Submission packaged and delivered

---

## Phase 30 — Gap Fixes (PRD Audit)

> These items were identified as missing after a critical PRD-vs-TODO audit.

### Gap 1 — C Injection Strategy for RNN / LSTM (FR-003, FR-004)

- [ ] Document chosen C-injection strategy in `docs/PRD_rnn.md` and `docs/PRD_lstm.md`: prepend C as the first time-step OR broadcast C to every time-step as extra features
- [ ] Decide input_size for RNN/LSTM: if C is concatenated per-step → input_size = 6 (1 window feature + 5 C); if C is prepended as step 0 → seq_len = 11, input_size = 1
- [ ] Add config key `"c_injection_strategy": "broadcast"` (or `"prepend"`) to `config/setup.json`
- [ ] Update `DataLoaderService` to build sequence tensors that include C for RNN/LSTM
- [ ] If strategy = broadcast: concatenate C to every time-step → shape (N, 10, 6); update seq DataLoader to use `x_seq_with_c`
- [ ] If strategy = prepend: prepend C as step-0 feature vector → shape (N, 11, 1); update seq DataLoader accordingly
- [ ] Update `RNNModel.__init__` to read `input_size` from config (not hardcoded to 1)
- [ ] Update `LSTMModel.__init__` to read `input_size` from config (not hardcoded to 1)
- [ ] Add test `test_seq_loader_batch_x_includes_C_features` — verify C values appear in batch tensor
- [ ] Add test `test_rnn_input_size_matches_c_injection_strategy` — input_size = 6 for broadcast or 1 for prepend
- [ ] Add test `test_lstm_input_size_matches_c_injection_strategy`
- [ ] Add test `test_rnn_output_changes_when_C_changes` — same window, different C → different prediction
- [ ] Add test `test_lstm_output_changes_when_C_changes`
- [ ] Add integration test `test_rnn_conditioning_on_C_lowers_mse_vs_no_C` — model with C beats model without C

### Gap 2 — Validation Split Contradiction (FR-005)

- [ ] Add TODO item to explicitly resolve val split: use X_val from dataset.npz (15%) as the validation set during training; do NOT re-split X_train
- [ ] Update `config/setup.json` `"val_split"` comment to clarify it refers to the pre-split ratio (15%), not 20%
- [ ] Add test `test_trainer_uses_provided_val_loader_not_internal_split` — trainer must not re-split train data
- [ ] Add test `test_val_loader_size_matches_dataset_npz_val_split` — size of val_loader ≈ 15% of total, not 20%
- [ ] Update `TrainingConfig` dataclass docstring to state val split comes from dataset.npz, not from trainer

### Gap 3 — Gradient Clipping for RNN (Risk Register)

- [ ] Add `"gradient_clip_norm": 1.0` key to `config/setup.json` training block (null means disabled)
- [ ] Add `gradient_clip_norm` field to `TrainingConfig` dataclass (Optional[float], default None)
- [ ] In `Trainer._run_epoch`: after `loss.backward()`, if `cfg.gradient_clip_norm` is not None call `torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)`
- [ ] Add test `test_gradient_clipping_applied_when_configured` — set high LR to produce large gradients, verify norms are clipped
- [ ] Add test `test_gradient_clipping_not_applied_when_config_is_null` — verify no clipping when config is None
- [ ] Add test `test_gradient_clip_norm_from_config` — verify trainer reads clip value from config, not hardcoded

### Gap 4 — DataLoader Reproducibility Seed (NFR-007)

- [ ] Set `torch.manual_seed(cfg.random_seed)` before creating DataLoaders in `DataLoaderService`
- [ ] Pass `generator=torch.Generator().manual_seed(seed)` to train DataLoader for reproducible shuffle
- [ ] Add test `test_train_loader_shuffle_reproducible_with_same_seed` — same seed → same first batch on two runs
- [ ] Add test `test_train_loader_shuffle_differs_across_epochs` — consecutive epochs produce different order (shuffle active)
- [ ] Add integration test `test_pipeline_produces_identical_output_with_same_seed` — run twice, compare train/val MSE

### Gap 5 — `evaluate_all()` Cold-Load from Checkpoint (FR-008)

- [ ] Add test `test_sdk_evaluate_all_works_without_in_memory_models` — delete in-memory models, call `evaluate_all()`, expect it to load from .pt files and produce MSE
- [ ] Add test `test_evaluate_all_loads_correct_checkpoint_per_model` — FCN eval uses fcn_best.pt, RNN uses rnn_best.pt, LSTM uses lstm_best.pt
- [ ] Add test `test_evaluate_all_raises_if_checkpoint_missing` — if .pt file absent, raise FileNotFoundError before evaluating
- [ ] In `sdk.py` `evaluate_all()`: instantiate fresh model objects, call `evaluator.load_checkpoint(model, path)` for each before evaluating

### Gap 6 — Noisy Signal Line Style Verification (FR-007)

- [ ] Add test `test_plot_noisy_signal_line_is_dashed` — inspect matplotlib line objects for noisy line, verify `linestyle == '--'`
- [ ] Add test `test_plot_clean_signal_line_is_solid` — verify clean line `linestyle == '-'`
- [ ] Add test `test_plot_line_colors_match_constants` — verify FCN=red, RNN=orange, LSTM=purple, clean=green, noisy=grey
- [ ] In `visualizer.py plot_clean_noisy_predicted`: explicitly pass `linestyle='--'` for noisy, `linestyle='-'` for clean

### Gap 7 — API Documentation Deliverable (Scope, Section 7)

- [ ] Create `docs/api_docs.md`
- [ ] Document `NeuralSignalSDK` public methods: `run_all()`, `train_model(name)`, `evaluate_all()`, `get_version()`
- [ ] Document each method: parameters, return type, raises, example usage
- [ ] Document `DataLoaderService.load()` public API
- [ ] Document `Preprocessor` public API: `fit()`, `transform()`, `fit_transform()`, `save_params()`, `load_params()`
- [ ] Document `Trainer.train()` public API
- [ ] Document `Evaluator.evaluate()` and `evaluate_all()` public API
- [ ] Document `Visualizer` public methods
- [ ] Add link to `docs/api_docs.md` in `README.md`
- [ ] Add `api_docs.md` to deliverables checklist in Phase 29

### Gap 8 — `training_log_<model>.csv` Path Naming (FR-005)

- [ ] Add test `test_sdk_passes_correct_log_path_for_fcn` — log path ends with `training_log_fcn.csv`
- [ ] Add test `test_sdk_passes_correct_log_path_for_rnn` — log path ends with `training_log_rnn.csv`
- [ ] Add test `test_sdk_passes_correct_log_path_for_lstm` — log path ends with `training_log_lstm.csv`
- [ ] Add `"training_log_fcn": "results/training_log_fcn.csv"` to output block in setup.json
- [ ] Add `"training_log_rnn": "results/training_log_rnn.csv"` to output block in setup.json
- [ ] Add `"training_log_lstm": "results/training_log_lstm.csv"` to output block in setup.json
- [ ] Update `OutputConfig` dataclass to include `training_log_fcn`, `training_log_rnn`, `training_log_lstm` fields
- [ ] In `sdk.py train_model()`: pass `cfg.output.training_log_{model_name}` as log_path to Trainer

### Gap 9 — Git Branching, PRs, and Version Tagging (CLAUDE.md)

- [ ] Create feature branch `feature/config-setup` before Phase 4 work
- [ ] Create feature branch `feature/shared-infra` before Phases 6-11
- [ ] Create feature branch `feature/data-pipeline` before Phases 12-13
- [ ] Create feature branch `feature/fcn-model` before Phase 14
- [ ] Create feature branch `feature/rnn-model` before Phase 15
- [ ] Create feature branch `feature/lstm-model` before Phase 16
- [ ] Create feature branch `feature/training-eval` before Phases 17-18
- [ ] Create feature branch `feature/visualization` before Phase 19
- [ ] Create feature branch `feature/sdk` before Phase 20
- [ ] Open PR for each feature branch — review before merging to master
- [ ] Tag `v1.0` on master after all phases complete and tests pass
- [ ] Add tagging step to Phase 29 final checklist

### Gap 10 — Device Handling (CPU / CUDA) in Trainer (NFR-008)

- [ ] Add `"device": "cpu"` key to `config/setup.json` training block (override via `TORCH_DEVICE` env var)
- [ ] Add `device` field to `TrainingConfig` dataclass
- [ ] In `Trainer.train()`: call `model.to(self._device)` before training loop
- [ ] In `Trainer._run_epoch()`: call `x.to(self._device)` and `y.to(self._device)` for every batch
- [ ] In `Evaluator.evaluate()`: move model and batches to correct device before inference
- [ ] Add test `test_trainer_moves_model_to_device` — verify model parameters are on expected device
- [ ] Add test `test_trainer_moves_batches_to_device` — verify batch tensors are on same device as model
- [ ] Add test `test_trainer_device_from_config` — verify device string read from config, not hardcoded
- [ ] Add test `test_training_runs_on_cpu` — explicitly set device=cpu, full train loop completes
- [ ] Verify `os.environ.get("TORCH_DEVICE", cfg.device)` used so env var overrides config

### Gap 11 — Validation MSE Divergence Check (Acceptance Criterion 2)

- [ ] Add integration test `test_val_mse_does_not_increase_unboundedly_for_fcn` — over all logged epochs, val_mse never increases by > 10× from best_val_mse
- [ ] Add integration test `test_val_mse_does_not_increase_unboundedly_for_rnn`
- [ ] Add integration test `test_val_mse_does_not_increase_unboundedly_for_lstm`
- [ ] Add unit test `test_early_stopping_prevents_val_mse_from_diverging` — inject mock val losses that grow; verify training stops within patience epochs

### Gap 12 — OAT Sensitivity TDD and Per-Parameter File Checks (FR-009, Phase 25)

- [ ] Create `src/neural_signal/services/sensitivity.py` for OAT sweep logic
- [ ] Add module docstring to `sensitivity.py`
- [ ] Define `SensitivityResult` dataclass: parameter_name, parameter_value, model_name, val_mse
- [ ] Define `SensitivityAnalyzer` class with docstring
- [ ] Implement `__init__(self, sdk: NeuralSignalSDK, output_dir: Path)`
- [ ] Implement `sweep(self, param_name: str, values: list) -> list[SensitivityResult]` — for each value, override config, train all 3 models, record val MSE
- [ ] Implement `save_results(self, results: list[SensitivityResult]) -> Path` — saves to `sensitivity_results.csv`
- [ ] Implement `plot_line_chart(self, param_name, results)` — line chart val MSE vs param value
- [ ] Implement `plot_heatmap(self, results)` — heatmap model × param value → val MSE
- [ ] Create `tests/unit/test_sensitivity.py` (TDD RED phase)
- [ ] Write `test_sensitivity_analyzer_instantiates`
- [ ] Write `test_sweep_returns_list_of_sensitivity_results`
- [ ] Write `test_sweep_result_has_parameter_name`
- [ ] Write `test_sweep_result_has_parameter_value`
- [ ] Write `test_sweep_result_has_model_name`
- [ ] Write `test_sweep_result_has_val_mse`
- [ ] Write `test_save_results_creates_csv`
- [ ] Write `test_csv_has_correct_columns`
- [ ] Write `test_plot_line_chart_creates_file`
- [ ] Write `test_plot_heatmap_creates_file`
- [ ] Run RED — confirm all FAIL
- [ ] Implement sensitivity.py — run GREEN — confirm all PASS
- [ ] Run REFACTOR — ruff check → 0 errors
- [ ] Add `test_run_all_creates_sensitivity_window_size_png` to integration tests
- [ ] Add `test_run_all_creates_sensitivity_hidden_size_png` to integration tests
- [ ] Add `test_run_all_creates_sensitivity_learning_rate_png` to integration tests
- [ ] Add `test_run_all_creates_sensitivity_dropout_rate_png` to integration tests
- [ ] Add `test_run_all_creates_sensitivity_batch_size_png` to integration tests
- [ ] Add `test_run_all_creates_sensitivity_results_csv` to integration tests
- [ ] Add `test_run_all_creates_sensitivity_heatmap_png` to integration tests
- [ ] Verify `sensitivity.py` ≤ 150 lines (split if needed)
- [ ] Add `docs/api_docs.md` to final submission checklist in Phase 29
- [ ] while you workind update the todo list to mark each task as done by  [x]
---

## Blocked / Notes

- Dataset (`data/dataset.npz`) must exist from Phase 1 before coding begins
- Supervisor approval required before Phase 1 (environment setup) coding begins
- `results/` and model checkpoints are git-ignored; regenerate locally with `uv run python -m neural_signal`
- PyTorch version must be ≥ 2.0 for `torch.manual_seed` and `DataLoader` compatibility
- If GPU unavailable, training will use CPU — configure `TORCH_DEVICE=cpu` in `.env`
- Gap 1 (C injection) must be resolved and approved before implementing DataLoaderService sequence path — the choice of strategy affects input_size of RNN/LSTM
- Gap 2 (val split 15% vs 20%) must be resolved before implementing Trainer — use X_val from dataset.npz
