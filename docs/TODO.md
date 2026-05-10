# TODO — Dataset Generation Phase

**Version:** 1.00
**Date:** 2026-05-10
**Status:** Not started — awaiting doc approval

Legend: `[ ]` = pending · `[~]` = in progress · `[x]` = done

---

## Phase 0 — Documentation & Approval

- [ ] Read PRD.md in full and verify signal definitions match specification
- [ ] Verify s1 formula: 2.0 · sin(2π · 5 · t) in PRD.md
- [ ] Verify s2 formula: 1.5 · sin(2π · 15 · t + π/4) in PRD.md
- [ ] Verify s3 formula: 0.8 · sin(2π · 50 · t) in PRD.md
- [ ] Verify s4 formula: 0.3 · sin(2π · 100 · t) in PRD.md
- [ ] Verify s5 = s1 + s2 + s3 + s4 stated clearly in PRD.md
- [ ] Verify noise model equation in PRD.md: `(A + N_amp) · sin(2πft + φ + N_phase)`
- [ ] Verify Gaussian white noise term documented in PRD.md
- [ ] Verify burst noise term documented in PRD.md
- [ ] Verify context window size W=10 documented in PRD.md
- [ ] Verify one-hot vector C definition in PRD.md
- [ ] Verify C is one-hot of length **5** (signal selector, NOT element-wise mask on window samples) in PRD.md
- [ ] Verify model-input concatenation documented in PRD.md: `x_input = concat([x_w, C])` → shape `(W+5,)` = `(15,)`
- [ ] Verify label definition (clean signal at window center) in PRD.md
- [ ] Verify train/val/test split ratios (70/15/15) in PRD.md
- [ ] Verify config schema documented in PRD.md
- [ ] Verify acceptance criteria complete in PRD.md
- [ ] Read PLAN.md in full and verify module responsibilities
- [ ] Verify DatasetSDK as single entry point in PLAN.md
- [ ] Verify all 5 services listed in PLAN.md
- [ ] Verify ApiGatekeeper in PLAN.md
- [ ] Verify data flow diagram correct in PLAN.md
- [ ] Verify noise model math in PLAN.md matches PRD.md
- [ ] Verify file size budget all ≤ 150 lines in PLAN.md
- [ ] Verify testing strategy lists all test files in PLAN.md
- [ ] Verify 5 visualisations planned in PLAN.md
- [ ] Read TODO.md (this file) and verify all phases present
- [ ] PRD.md submitted to supervisor for approval
- [ ] PLAN.md submitted to supervisor for approval
- [ ] TODO.md submitted to supervisor for approval
- [ ] PRD_dataset_generation.md drafted (per-algorithm PRD)
- [ ] PRD_dataset_generation.md submitted for approval
- [ ] Supervisor approval received on PRD.md
- [ ] Supervisor approval received on PLAN.md
- [ ] Supervisor approval received on PRD_dataset_generation.md
- [ ] Supervisor approval received on TODO.md
- [ ] All approvals recorded with date and version

---

## Phase 1 — Environment Setup

- [ ] Verify Python 3.13 is installed (`python --version`)
- [ ] Verify uv is installed (`uv --version`)
- [ ] Verify ruff is available (`ruff --version`)
- [ ] Run `uv init signal_dataset` to initialise project
- [ ] Confirm `pyproject.toml` created by uv init
- [ ] Confirm `uv.lock` created after init
- [ ] Run `uv add numpy` and verify added to `pyproject.toml`
- [ ] Run `uv add matplotlib` and verify added
- [ ] Run `uv add seaborn` and verify added
- [ ] Run `uv add plotly` and verify added
- [ ] Run `uv add jupyter` and verify added
- [ ] Run `uv add scipy` and verify added (for FFT / statistical tests)
- [ ] Run `uv add --dev pytest` and verify added to dev deps
- [ ] Run `uv add --dev pytest-cov` and verify added
- [ ] Run `uv add --dev ruff` and verify added
- [ ] Run `uv add --dev pytest-mock` and verify added
- [ ] Run `uv lock` and verify `uv.lock` updated
- [ ] Run `uv sync` and verify all packages install without errors
- [ ] Confirm no `requirements.txt` exists (forbidden)
- [ ] Confirm no `pip install` commands used anywhere

---

## Phase 2 — Folder Structure Creation

- [ ] Create `src/` directory
- [ ] Create `src/signal_dataset/` directory
- [ ] Create `src/signal_dataset/sdk/` directory
- [ ] Create `src/signal_dataset/services/` directory
- [ ] Create `src/signal_dataset/shared/` directory
- [ ] Create `tests/` directory
- [ ] Create `tests/unit/` directory
- [ ] Create `tests/integration/` directory
- [ ] Create `config/` directory
- [ ] Create `data/` directory
- [ ] Create `results/` directory
- [ ] Create `assets/` directory
- [ ] Create `notebooks/` directory
- [ ] Create `docs/` directory (already exists)
- [ ] Verify full tree matches PLAN.md exactly

---

## Phase 3 — pyproject.toml Configuration

- [ ] Set `[project] name = "signal-dataset"`
- [ ] Set `[project] version = "1.0.0"`
- [ ] Set `[project] requires-python = ">=3.10"`
- [ ] Add `numpy` to `[project] dependencies`
- [ ] Add `matplotlib` to `[project] dependencies`
- [ ] Add `seaborn` to `[project] dependencies`
- [ ] Add `plotly` to `[project] dependencies`
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
- [ ] Add `[tool.coverage.run] omit = ["src/main.py", "*/tests/*"]`
- [ ] Add `[tool.coverage.report] fail_under = 85`
- [ ] Add `[build-system]` table with hatchling or setuptools
- [ ] Add `[project.scripts] signal-dataset = "signal_dataset.main:main"`
- [ ] Verify `uv run ruff check pyproject.toml` passes
- [ ] Verify `uv sync` still works after all edits

---

## Phase 4 — Configuration Files

- [ ] Create `config/setup.json`
- [ ] Add `"dataset"` block to `setup.json`
- [ ] Add `"duration_sec": 10` to dataset block
- [ ] Add `"sample_rate_hz": 1000` to dataset block
- [ ] Add `"window_size": 10` to dataset block
- [ ] Add `"train_ratio": 0.70` to dataset block
- [ ] Add `"val_ratio": 0.15` to dataset block
- [ ] Add `"test_ratio": 0.15` to dataset block
- [ ] Add `"random_seed": 42` to dataset block
- [ ] Add `"signals"` array to `setup.json`
- [ ] Add s1 entry: `{"id":"s1","amplitude":2.0,"frequency_hz":5,"phase_rad":0}`
- [ ] Add s2 entry: `{"id":"s2","amplitude":1.5,"frequency_hz":15,"phase_rad":0.7854}`
- [ ] Add s3 entry: `{"id":"s3","amplitude":0.8,"frequency_hz":50,"phase_rad":0}`
- [ ] Add s4 entry: `{"id":"s4","amplitude":0.3,"frequency_hz":100,"phase_rad":0}`
- [ ] Add `"noise"` block to `setup.json`
- [ ] Add `"gaussian"` sub-block under noise
- [ ] Add `"sigma_amp": [0.05, 0.05, 0.03, 0.02]` to gaussian block
- [ ] Add `"sigma_phase": [0.05, 0.05, 0.03, 0.02]` to gaussian block
- [ ] Add `"burst"` sub-block under noise
- [ ] Add `"probability": 0.01` to burst block
- [ ] Add `"duration_range_samples": [5, 20]` to burst block
- [ ] Add `"amp_magnitude": 0.5` to burst block
- [ ] Add `"phase_magnitude": 0.3` to burst block
- [ ] Add `"output"` block to `setup.json`
- [ ] Add `"dataset_path": "data/dataset.npz"` to output block
- [ ] Add `"raw_signals_path": "data/signals_raw.npz"` to output block
- [ ] Add `"results_dir": "results/"` to output block
- [ ] Validate `setup.json` is valid JSON (`python -m json.tool config/setup.json`)
- [ ] Create `config/rate_limits.json`
- [ ] Add `"rate_limits"` top-level key
- [ ] Add `"version": "1.00"` inside rate_limits
- [ ] Add `"services"` block inside rate_limits
- [ ] Add `"default"` service with `"requests_per_minute": 30`
- [ ] Add `"requests_per_hour": 500` to default service
- [ ] Add `"concurrent_max": 5` to default service
- [ ] Add `"retry_after_seconds": 30` to default service
- [ ] Add `"max_retries": 3` to default service
- [ ] Add `"queue_max_depth": 100` to default service
- [ ] Validate `rate_limits.json` is valid JSON
- [ ] Confirm no numeric literals appear in any `.py` file that should come from config

---

## Phase 5 — .gitignore and .env-example

- [ ] Create `.gitignore`
- [ ] Add `.env` to `.gitignore`
- [ ] Add `*.key` to `.gitignore`
- [ ] Add `*.pem` to `.gitignore`
- [ ] Add `credentials.json` to `.gitignore`
- [ ] Add `data/` to `.gitignore`
- [ ] Add `results/` to `.gitignore`
- [ ] Add `__pycache__/` to `.gitignore`
- [ ] Add `*.pyc` to `.gitignore`
- [ ] Add `.pytest_cache/` to `.gitignore`
- [ ] Add `.coverage` to `.gitignore`
- [ ] Add `htmlcov/` to `.gitignore`
- [ ] Add `.ruff_cache/` to `.gitignore`
- [ ] Add `*.egg-info/` to `.gitignore`
- [ ] Add `.venv/` to `.gitignore`
- [ ] Add `dist/` to `.gitignore`
- [ ] Add `build/` to `.gitignore`
- [ ] Add `*.ipynb_checkpoints` to `.gitignore`
- [ ] Create `.env-example`
- [ ] Add `API_KEY=your_api_key_here` (dummy) to `.env-example`
- [ ] Add `LOG_LEVEL=INFO` to `.env-example`
- [ ] Add `DATA_DIR=data/` to `.env-example`
- [ ] Add `RESULTS_DIR=results/` to `.env-example`
- [ ] Verify `.env` is not committed
- [ ] Verify `.env-example` is committed

---

## Phase 6 — `src/signal_dataset/shared/version.py`

- [ ] Create file `src/signal_dataset/shared/version.py`
- [ ] Add module docstring explaining version tracking purpose
- [ ] Define `__version__ = "1.00"`
- [ ] Define `VERSION_MAJOR = 1`
- [ ] Define `VERSION_MINOR = 0`
- [ ] Define `BUILD_DATE = "2026-05-10"`
- [ ] Define `get_version()` function returning `__version__`
- [ ] Add docstring to `get_version()` explaining return value
- [ ] Verify file is ≤ 150 lines
- [ ] Verify `uv run ruff check src/signal_dataset/shared/version.py` → 0 errors

---

## Phase 7 — `src/signal_dataset/constants.py`

- [ ] Create file `src/signal_dataset/constants.py`
- [ ] Add module docstring explaining this is the single source for all physical/math constants
- [ ] Define `TWO_PI = 2.0 * math.pi`
- [ ] Define `DEFAULT_DURATION_SEC = 10` (fallback only, config takes priority)
- [ ] Define `DEFAULT_SAMPLE_RATE_HZ = 1000`
- [ ] Define `DEFAULT_WINDOW_SIZE = 10`
- [ ] Define `N_SIGNALS = 4` (number of base sine components)
- [ ] Define `N_TOTAL_SIGNALS = 5` (including composite)
- [ ] Define `COMPOSITE_SIGNAL_ID = "s5"`
- [ ] Define `SIGNAL_IDS = ["s1", "s2", "s3", "s4", "s5"]`
- [ ] Define `DATASET_NPZ_KEY_X_TRAIN = "X_train"`
- [ ] Define `DATASET_NPZ_KEY_X_VAL = "X_val"`
- [ ] Define `DATASET_NPZ_KEY_X_TEST = "X_test"`
- [ ] Define `DATASET_NPZ_KEY_C_TRAIN = "C_train"` — one-hot signal selector
- [ ] Define `DATASET_NPZ_KEY_C_VAL = "C_val"`
- [ ] Define `DATASET_NPZ_KEY_C_TEST = "C_test"`
- [ ] Define `DATASET_NPZ_KEY_Y_TRAIN = "y_train"`
- [ ] Define `DATASET_NPZ_KEY_Y_VAL = "y_val"`
- [ ] Define `DATASET_NPZ_KEY_Y_TEST = "y_test"`
- [ ] Define `N_SELECTOR_CLASSES = 5` — length of C vector (one per signal)
- [ ] Define `RAW_NPZ_KEY_CLEAN = "clean_signals"`
- [ ] Define `RAW_NPZ_KEY_NOISY = "noisy_signals"`
- [ ] Define `RAW_NPZ_KEY_TIME = "time_axis"`
- [ ] Define `NOISE_TYPE_GAUSSIAN = "gaussian"`
- [ ] Define `NOISE_TYPE_BURST = "burst"`
- [ ] Import only `math` (no numpy in constants)
- [ ] Verify file is ≤ 150 lines
- [ ] Verify `uv run ruff check src/signal_dataset/constants.py` → 0 errors
- [ ] Verify no business logic in this file — constants only

---

## Phase 8 — `src/signal_dataset/__init__.py`

- [ ] Create `src/signal_dataset/__init__.py`
- [ ] Add module-level docstring describing the package
- [ ] Import `__version__` from `shared.version`
- [ ] Define `__all__` list: `["DatasetSDK", "__version__"]`
- [ ] Import `DatasetSDK` from `sdk.sdk`
- [ ] Verify file is ≤ 150 lines
- [ ] Verify `uv run ruff check src/signal_dataset/__init__.py` → 0 errors
- [ ] Create `src/signal_dataset/sdk/__init__.py` (empty, or minimal)
- [ ] Create `src/signal_dataset/services/__init__.py` (empty, or minimal)
- [ ] Create `src/signal_dataset/shared/__init__.py` (empty, or minimal)

---

## Phase 9 — `tests/conftest.py` (Shared Fixtures)

- [ ] Create `tests/conftest.py`
- [ ] Add module docstring explaining shared fixtures
- [ ] Import `pytest`, `numpy`, `pathlib.Path`
- [ ] Define `config_path` fixture returning path to `config/setup.json`
- [ ] Define `rate_limits_path` fixture returning path to `config/rate_limits.json`
- [ ] Define `sample_config_dict` fixture with minimal valid config dict (no file I/O)
- [ ] Define `time_axis_fixture` fixture: `np.linspace(0, 10, 10000, endpoint=False)`
- [ ] Define `small_time_axis` fixture: 100-sample axis for fast tests
- [ ] Define `clean_s1_fixture` — precomputed s1 for small axis
- [ ] Define `clean_s2_fixture` — precomputed s2 for small axis
- [ ] Define `clean_s3_fixture` — precomputed s3 for small axis
- [ ] Define `clean_s4_fixture` — precomputed s4 for small axis
- [ ] Define `clean_s5_fixture` — sum of the four above
- [ ] Define `noisy_signals_fixture` — dict with 5 noisy arrays (small axis)
- [ ] Define `tmp_data_dir` fixture using `tmp_path` for isolated file I/O
- [ ] Define `window_size_fixture` returning 10
- [ ] Define `random_seed_fixture` returning 42
- [ ] Create `tests/unit/__init__.py` (empty)
- [ ] Create `tests/integration/__init__.py` (empty)
- [ ] Verify conftest.py is ≤ 150 lines
- [ ] Verify `uv run ruff check tests/conftest.py` → 0 errors

---

## Phase 10 — `shared/config.py` — TDD RED Phase

- [ ] Create `tests/unit/test_config.py`
- [ ] Add module docstring to test file
- [ ] Write `test_config_manager_loads_valid_json` — ConfigManager loads setup.json without error
- [ ] Write `test_config_manager_returns_dataset_section` — `cfg.dataset` not None
- [ ] Write `test_config_duration_sec_equals_10` — `cfg.dataset.duration_sec == 10`
- [ ] Write `test_config_sample_rate_hz_equals_1000` — `cfg.dataset.sample_rate_hz == 1000`
- [ ] Write `test_config_window_size_equals_10` — `cfg.dataset.window_size == 10`
- [ ] Write `test_config_train_ratio_equals_0_70` — `cfg.dataset.train_ratio == 0.70`
- [ ] Write `test_config_val_ratio_equals_0_15`
- [ ] Write `test_config_test_ratio_equals_0_15`
- [ ] Write `test_config_split_ratios_sum_to_1` — train+val+test == 1.0
- [ ] Write `test_config_random_seed_equals_42`
- [ ] Write `test_config_signals_list_has_4_entries` — len(cfg.signals) == 4
- [ ] Write `test_config_s1_amplitude_equals_2_0`
- [ ] Write `test_config_s1_frequency_equals_5`
- [ ] Write `test_config_s1_phase_equals_0`
- [ ] Write `test_config_s2_amplitude_equals_1_5`
- [ ] Write `test_config_s2_frequency_equals_15`
- [ ] Write `test_config_s2_phase_approx_pi_over_4`
- [ ] Write `test_config_s3_amplitude_equals_0_8`
- [ ] Write `test_config_s3_frequency_equals_50`
- [ ] Write `test_config_s3_phase_equals_0`
- [ ] Write `test_config_s4_amplitude_equals_0_3`
- [ ] Write `test_config_s4_frequency_equals_100`
- [ ] Write `test_config_s4_phase_equals_0`
- [ ] Write `test_config_s2_phase_equals_0_7854` — exact value 0.7854 rad (π/4)
- [ ] Write `test_config_noise_gaussian_sigma_amp_has_4_entries`
- [ ] Write `test_config_noise_gaussian_sigma_amp_values_exact` — [0.05, 0.05, 0.03, 0.02]
- [ ] Write `test_config_noise_gaussian_sigma_amp_s1_equals_0_05`
- [ ] Write `test_config_noise_gaussian_sigma_amp_s2_equals_0_05`
- [ ] Write `test_config_noise_gaussian_sigma_amp_s3_equals_0_03`
- [ ] Write `test_config_noise_gaussian_sigma_amp_s4_equals_0_02`
- [ ] Write `test_config_noise_gaussian_sigma_phase_has_4_entries`
- [ ] Write `test_config_noise_gaussian_sigma_phase_values_exact` — [0.05, 0.05, 0.03, 0.02]
- [ ] Write `test_config_noise_gaussian_sigma_phase_s1_equals_0_05`
- [ ] Write `test_config_noise_gaussian_sigma_phase_s2_equals_0_05`
- [ ] Write `test_config_noise_gaussian_sigma_phase_s3_equals_0_03`
- [ ] Write `test_config_noise_gaussian_sigma_phase_s4_equals_0_02`
- [ ] Write `test_config_noise_burst_probability_equals_0_01` — exact value, not just positive
- [ ] Write `test_config_noise_burst_duration_min_equals_5` — exact lower bound
- [ ] Write `test_config_noise_burst_duration_max_equals_20` — exact upper bound
- [ ] Write `test_config_noise_burst_duration_range_valid` — min < max, both positive
- [ ] Write `test_config_noise_burst_amp_magnitude_equals_0_5` — exact value
- [ ] Write `test_config_noise_burst_phase_magnitude_equals_0_3` — exact value
- [ ] Write `test_config_raises_on_missing_file` — FileNotFoundError on bad path
- [ ] Write `test_config_raises_on_invalid_json` — JSONDecodeError or ConfigValidationError
- [ ] Write `test_config_raises_on_negative_duration`
- [ ] Write `test_config_raises_on_zero_sample_rate`
- [ ] Write `test_config_raises_on_window_size_zero`
- [ ] Write `test_config_raises_on_split_ratios_not_summing_to_1`
- [ ] Write `test_config_raises_on_negative_amplitude`
- [ ] Write `test_config_raises_on_negative_frequency`
- [ ] Write `test_config_raises_on_empty_signals_list`
- [ ] Write `test_config_raises_on_negative_noise_sigma`
- [ ] Write `test_config_raises_on_burst_probability_above_1`
- [ ] Write `test_config_loads_rate_limits_json`
- [ ] Write `test_config_rate_limit_default_service_exists`
- [ ] Write `test_config_rate_limit_requests_per_minute_positive`
- [ ] Verify test file is ≤ 150 lines (split into `test_config_signals.py` if needed)
- [ ] Run tests — confirm all FAIL (RED state confirmed)

---

## Phase 10 — `shared/config.py` — GREEN Phase

- [ ] Create `src/signal_dataset/shared/config.py`
- [ ] Add module docstring explaining config loading and validation
- [ ] Import `json`, `os`, `pathlib`, `dataclasses`, `typing`
- [ ] Define `SignalConfig` dataclass: `id`, `amplitude`, `frequency_hz`, `phase_rad`
- [ ] Add docstring to `SignalConfig`
- [ ] Define `GaussianNoiseConfig` dataclass: `sigma_amp`, `sigma_phase`
- [ ] Add docstring to `GaussianNoiseConfig`
- [ ] Define `BurstNoiseConfig` dataclass: `probability`, `duration_range_samples`, `amp_magnitude`, `phase_magnitude`
- [ ] Add docstring to `BurstNoiseConfig`
- [ ] Define `NoiseConfig` dataclass: `gaussian`, `burst`
- [ ] Add docstring to `NoiseConfig`
- [ ] Define `DatasetConfig` dataclass: `duration_sec`, `sample_rate_hz`, `window_size`, `train_ratio`, `val_ratio`, `test_ratio`, `random_seed`
- [ ] Add docstring to `DatasetConfig`
- [ ] Define `OutputConfig` dataclass: `dataset_path`, `raw_signals_path`, `results_dir`
- [ ] Add docstring to `OutputConfig`
- [ ] Define `AppConfig` dataclass: `dataset`, `signals`, `noise`, `output`
- [ ] Add docstring to `AppConfig`
- [ ] Define `ConfigValidationError(Exception)` custom exception
- [ ] Define `ConfigManager` class
- [ ] Add class-level docstring to `ConfigManager`
- [ ] Implement `ConfigManager.__init__(self, config_path, rate_limits_path)`
- [ ] Implement `ConfigManager.load() -> AppConfig`
- [ ] Implement private `_load_json(path) -> dict` with error handling
- [ ] Implement private `_parse_dataset(raw) -> DatasetConfig`
- [ ] Implement private `_parse_signals(raw) -> list[SignalConfig]`
- [ ] Implement private `_parse_noise(raw) -> NoiseConfig`
- [ ] Implement private `_parse_output(raw) -> OutputConfig`
- [ ] Implement private `_validate(config: AppConfig)` raising `ConfigValidationError`
- [ ] Validate `duration_sec > 0`
- [ ] Validate `sample_rate_hz > 0`
- [ ] Validate `window_size > 0`
- [ ] Validate `train_ratio + val_ratio + test_ratio ≈ 1.0` (tolerance 1e-9)
- [ ] Validate all ratios in (0, 1)
- [ ] Validate all signal amplitudes > 0
- [ ] Validate all signal frequencies > 0
- [ ] Validate sigma_amp all ≥ 0
- [ ] Validate sigma_phase all ≥ 0
- [ ] Validate burst probability in [0, 1]
- [ ] Validate burst duration_range_samples[0] < duration_range_samples[1]
- [ ] Validate len(sigma_amp) == len(signals)
- [ ] Verify file is ≤ 150 lines (split into `config_models.py` for dataclasses if needed)
- [ ] Run tests — confirm all PASS (GREEN state confirmed)

---

## Phase 10 — `shared/config.py` — REFACTOR Phase

- [ ] Check for any duplicated parsing logic — extract helpers
- [ ] Ensure each method is single-responsibility
- [ ] Ensure all error messages are descriptive (include field name and bad value)
- [ ] Remove any dead code or commented-out lines
- [ ] Confirm ≤ 150 lines total
- [ ] Run `uv run ruff check src/signal_dataset/shared/config.py` → 0 errors
- [ ] Re-run tests — still PASS after refactor

---

## Phase 11 — `shared/gatekeeper.py` — TDD RED Phase

- [ ] Create `tests/unit/test_gatekeeper.py`
- [ ] Add module docstring to test file
- [ ] Write `test_gatekeeper_instantiates_with_valid_config`
- [ ] Write `test_gatekeeper_execute_calls_function`
- [ ] Write `test_gatekeeper_execute_returns_function_result`
- [ ] Write `test_gatekeeper_execute_logs_call` — check call is recorded in log
- [ ] Write `test_gatekeeper_execute_single_callable_no_error`
- [ ] Write `test_gatekeeper_queue_status_returns_queue_status_object`
- [ ] Write `test_gatekeeper_queue_status_depth_starts_at_0`
- [ ] Write `test_gatekeeper_queue_status_processed_starts_at_0`
- [ ] Write `test_gatekeeper_queue_status_failed_starts_at_0`
- [ ] Write `test_gatekeeper_get_call_log_returns_list`
- [ ] Write `test_gatekeeper_get_call_log_is_empty_initially`
- [ ] Write `test_gatekeeper_get_call_log_grows_after_execute`
- [ ] Write `test_gatekeeper_call_log_entry_has_timestamp`
- [ ] Write `test_gatekeeper_call_log_entry_has_function_name`
- [ ] Write `test_gatekeeper_call_log_entry_has_status_success`
- [ ] Write `test_gatekeeper_call_log_entry_has_status_failure_on_exception`
- [ ] Write `test_gatekeeper_execute_retries_on_exception`
- [ ] Write `test_gatekeeper_execute_raises_after_max_retries`
- [ ] Write `test_gatekeeper_execute_respects_max_retries_config`
- [ ] Write `test_gatekeeper_queue_depth_not_exceeded` — backpressure logic
- [ ] Write `test_gatekeeper_raises_when_queue_full`
- [ ] Verify test file is ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

---

## Phase 11 — `shared/gatekeeper.py` — GREEN Phase

- [ ] Create `src/signal_dataset/shared/gatekeeper.py`
- [ ] Add module docstring explaining centralised API call management
- [ ] Import `time`, `queue`, `threading`, `dataclasses`, `typing`, `logging`
- [ ] Define `RateLimitConfig` dataclass: `requests_per_minute`, `requests_per_hour`, `concurrent_max`, `retry_after_seconds`, `max_retries`, `queue_max_depth`
- [ ] Add docstring to `RateLimitConfig`
- [ ] Define `QueueStatus` dataclass: `depth`, `processed`, `failed`
- [ ] Add docstring to `QueueStatus`
- [ ] Define `CallLogEntry` dataclass: `timestamp`, `function_name`, `status`, `error_message`
- [ ] Add docstring to `CallLogEntry`
- [ ] Define `ApiGatekeeper` class
- [ ] Add class-level docstring
- [ ] Implement `__init__(self, config: RateLimitConfig)`
- [ ] Initialise FIFO queue (`queue.Queue`) with maxsize from config
- [ ] Initialise `_call_log: list[CallLogEntry]`
- [ ] Initialise counters: `_processed`, `_failed`
- [ ] Implement `execute(self, api_call, *args, **kwargs)` with retry loop
- [ ] Implement retry logic: up to `max_retries`, sleep `retry_after_seconds` between
- [ ] Log each attempt to `_call_log`
- [ ] Raise `GatekeeperQueueFullError` when queue at max depth
- [ ] Implement `get_queue_status(self) -> QueueStatus`
- [ ] Implement `get_call_log(self) -> list[CallLogEntry]`
- [ ] Define `GatekeeperQueueFullError(Exception)`
- [ ] Define `GatekeeperMaxRetriesError(Exception)`
- [ ] Verify file is ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

---

## Phase 11 — `shared/gatekeeper.py` — REFACTOR Phase

- [ ] Extract retry logic into private `_execute_with_retry` method
- [ ] Extract log-entry creation into private `_log_call` method
- [ ] Ensure thread safety — review lock usage
- [ ] Confirm all docstrings present
- [ ] Run `uv run ruff check src/signal_dataset/shared/gatekeeper.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 12 — `services/signal_generator.py` — TDD RED Phase

- [ ] Create `tests/unit/test_signal_generator.py`
- [ ] Add module docstring
- [ ] Write `test_signal_generator_instantiates`
- [ ] Write `test_generate_time_axis_shape` — shape == (10000,)
- [ ] Write `test_generate_time_axis_start_equals_0` — t[0] == 0.0
- [ ] Write `test_generate_time_axis_last_sample_equals_9_999` — t[-1] == 9.999 exactly (endpoint=False)
- [ ] Write `test_generate_time_axis_end_less_than_duration` — t[-1] < 10.0
- [ ] Write `test_generate_time_axis_step_equals_1_over_sample_rate` — t[1]-t[0] == 0.001
- [ ] Write `test_generate_s1_shape` — (10000,)
- [ ] Write `test_generate_s2_shape`
- [ ] Write `test_generate_s3_shape`
- [ ] Write `test_generate_s4_shape`
- [ ] Write `test_generate_s5_shape`
- [ ] Write `test_generate_s1_max_amplitude_approx_2_0` — np.max(abs(s1)) ≈ 2.0
- [ ] Write `test_generate_s2_max_amplitude_approx_1_5`
- [ ] Write `test_generate_s3_max_amplitude_approx_0_8`
- [ ] Write `test_generate_s4_max_amplitude_approx_0_3`
- [ ] Write `test_generate_s1_is_zero_mean` — np.mean(s1) ≈ 0 (tolerance 0.01)
- [ ] Write `test_generate_s2_is_zero_mean`
- [ ] Write `test_generate_s3_is_zero_mean`
- [ ] Write `test_generate_s4_is_zero_mean`
- [ ] Write `test_generate_s5_equals_sum_of_components` — np.allclose(s5, s1+s2+s3+s4)
- [ ] Write `test_generate_s1_dominant_frequency_via_fft` — FFT peak at 5 Hz
- [ ] Write `test_generate_s2_dominant_frequency_via_fft` — FFT peak at 15 Hz
- [ ] Write `test_generate_s3_dominant_frequency_via_fft` — FFT peak at 50 Hz
- [ ] Write `test_generate_s4_dominant_frequency_via_fft` — FFT peak at 100 Hz
- [ ] Write `test_generate_s2_phase_offset` — verify π/4 phase via correlation
- [ ] Write `test_generate_all_returns_dict_with_5_keys`
- [ ] Write `test_generate_all_keys_are_s1_through_s5`
- [ ] Write `test_generate_all_values_are_ndarrays`
- [ ] Write `test_generate_no_nans_in_any_signal`
- [ ] Write `test_generate_no_infs_in_any_signal`
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

---

## Phase 12 — `services/signal_generator.py` — GREEN Phase

- [ ] Create `src/signal_dataset/services/signal_generator.py`
- [ ] Add module docstring
- [ ] Import `numpy as np` and relevant constants from `constants.py`
- [ ] Import `SignalConfig`, `DatasetConfig` from `shared.config`
- [ ] Define `SignalGenerator` class
- [ ] Add class-level docstring
- [ ] Implement `__init__(self, dataset_cfg: DatasetConfig, signals: list[SignalConfig])`
- [ ] Store `self._dataset_cfg` and `self._signals`
- [ ] Implement `generate_time_axis(self) -> np.ndarray`
- [ ] Use `np.linspace(0, duration, n_samples, endpoint=False)` — no hardcoded values
- [ ] Implement `generate_single(self, signal_cfg: SignalConfig, t: np.ndarray) -> np.ndarray`
- [ ] Formula: `A * np.sin(TWO_PI * f * t + phi)` using `TWO_PI` from constants
- [ ] Implement `generate_composite(self, components: dict[str, np.ndarray]) -> np.ndarray`
- [ ] Sum all component arrays (s1+s2+s3+s4)
- [ ] Implement `generate_all(self) -> dict[str, np.ndarray]`
- [ ] Call `generate_time_axis`, loop over signals, call `generate_single`, then `generate_composite`
- [ ] Return dict with keys `"s1"` through `"s5"` using `SIGNAL_IDS` from constants
- [ ] Verify file is ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

---

## Phase 12 — `services/signal_generator.py` — REFACTOR Phase

- [ ] Ensure no magic numbers (`2.0`, `1.5`, etc.) remain in source — all from config
- [ ] Check for code duplication across generate methods — extract if needed
- [ ] Confirm every method has a docstring
- [ ] Run `uv run ruff check src/signal_dataset/services/signal_generator.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 13 — `services/noise_injector.py` — TDD RED Phase

- [ ] Create `tests/unit/test_noise_injector.py`
- [ ] Add module docstring
- [ ] Write `test_noise_injector_instantiates`
- [ ] Write `test_generate_gaussian_amp_noise_shape` — shape == (10000,)
- [ ] Write `test_generate_gaussian_amp_noise_mean_near_zero` — abs(mean) < 3*sigma/sqrt(N)
- [ ] Write `test_generate_gaussian_amp_noise_std_near_sigma` — std ≈ sigma (tolerance 10%)
- [ ] Write `test_generate_gaussian_phase_noise_shape`
- [ ] Write `test_generate_gaussian_phase_noise_mean_near_zero`
- [ ] Write `test_generate_gaussian_phase_noise_std_near_sigma`
- [ ] Write `test_generate_burst_mask_is_binary` — all values in {0, 1}
- [ ] Write `test_generate_burst_mask_shape` — shape == (10000,)
- [ ] Write `test_generate_burst_mask_burst_durations_in_range` — check run-length of 1s
- [ ] Write `test_generate_burst_noise_amp_shape`
- [ ] Write `test_generate_burst_noise_phase_shape`
- [ ] Write `test_inject_single_differs_from_clean` — not allclose
- [ ] Write `test_inject_single_shape_preserved` — shape == clean signal shape
- [ ] Write `test_inject_single_no_nan`
- [ ] Write `test_inject_single_no_inf`
- [ ] Write `test_inject_all_returns_dict_with_5_keys`
- [ ] Write `test_inject_all_keys_match_signal_ids`
- [ ] Write `test_inject_all_values_are_ndarrays`
- [ ] Write `test_n_amp_and_n_phase_are_independent` — correlation between them near 0
- [ ] Write `test_reproducibility_with_same_seed` — same seed → same noisy signal
- [ ] Write `test_different_seeds_give_different_noise`
- [ ] Write `test_zero_sigma_gaussian_produces_no_gaussian_noise` — noisy == clean when sigma=0 and no burst
- [ ] Write `test_zero_burst_probability_produces_no_burst_events`
- [ ] Write `test_burst_magnitude_bounded` — burst values ≤ amp_magnitude
- [ ] Write `test_composite_noisy_differs_from_clean_composite`
- [ ] Write `test_inject_all_s5_noisy_equals_sum_of_s1_s2_s3_s4_noisy` — unit test: s5_noisy == s1_noisy + s2_noisy + s3_noisy + s4_noisy (allclose)
- [ ] Write `test_noise_formula_n_amp_is_additive_to_amplitude` — with constant N_amp offset and zero N_phase, verify peak amplitude shifts by offset
- [ ] Write `test_noise_formula_n_phase_is_additive_to_phase` — with constant N_phase offset and zero N_amp, verify phase shift in output
- [ ] Verify test file ≤ 150 lines (split if needed)
- [ ] Run tests — confirm all FAIL (RED)

---

## Phase 13 — `services/noise_injector.py` — GREEN Phase

- [ ] Create `src/signal_dataset/services/noise_injector.py`
- [ ] Add module docstring
- [ ] Import `numpy as np`, `constants`, `config` types
- [ ] Define `NoiseInjector` class
- [ ] Add class-level docstring
- [ ] Implement `__init__(self, noise_cfg: NoiseConfig, dataset_cfg: DatasetConfig, seed: int)`
- [ ] Store `self._rng = np.random.default_rng(seed)`
- [ ] Implement `_generate_gaussian_noise(self, sigma: float, n: int) -> np.ndarray`
- [ ] Implement `_generate_burst_mask(self, n: int) -> np.ndarray`
- [ ] Use Bernoulli trial per sample with burst probability; extend runs to duration
- [ ] Implement `_generate_burst_noise(self, mask: np.ndarray, magnitude: float) -> np.ndarray`
- [ ] Uniform random values in [-magnitude, +magnitude] where mask is 1
- [ ] Implement `_compute_n_amp(self, signal_idx: int, n: int) -> np.ndarray`
- [ ] Sum gaussian_amp + burst_amp
- [ ] Implement `_compute_n_phase(self, signal_idx: int, n: int) -> np.ndarray`
- [ ] Sum gaussian_phase + burst_phase
- [ ] Implement `inject_single(self, clean: np.ndarray, signal_cfg: SignalConfig, signal_idx: int) -> np.ndarray`
- [ ] Formula: `(A + N_amp) * np.sin(TWO_PI * f * t + phi + N_phase)` — read t from stored axis
- [ ] Implement `inject_all(self, clean_signals: dict, signal_configs: list, t: np.ndarray) -> dict`
- [ ] Loop over s1-s4, call `inject_single`, compute s5_noisy as sum of noisy components
- [ ] Verify file is ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

---

## Phase 13 — `services/noise_injector.py` — REFACTOR Phase

- [ ] Extract burst run-length extension into `_expand_burst_mask` helper
- [ ] Ensure `_generate_burst_mask` and `_generate_burst_noise` are independently testable
- [ ] Confirm no shared mutable state between inject calls
- [ ] Run `uv run ruff check src/signal_dataset/services/noise_injector.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 14 — `services/windower.py` — TDD RED Phase

- [ ] Create `tests/unit/test_windower.py`
- [ ] Add module docstring
- [ ] Write `test_windower_instantiates`
- [ ] Write `test_build_windows_X_shape_is_n_windows_by_W` — X has shape (N, W)
- [ ] Write `test_n_windows_equals_n_samples_minus_W_plus_1`
- [ ] Write `test_X_windows_sourced_from_s5_noisy_only` — X[k] == s5_noisy[k:k+W]
- [ ] Write `test_X_does_not_equal_s1_noisy_window` — confirm s1_noisy is NOT the source
- [ ] Write `test_X_does_not_equal_s2_noisy_window`
- [ ] Write `test_X_does_not_equal_s3_noisy_window`
- [ ] Write `test_X_does_not_equal_s4_noisy_window`
- [ ] Write `test_C_shape_is_n_windows_by_5` — C has shape (N, 5), NOT (N, W)
- [ ] Write `test_each_C_vector_is_one_hot` — np.sum(C[k]) == 1 for all k
- [ ] Write `test_each_C_vector_has_exactly_one_nonzero`
- [ ] Write `test_C_values_are_0_or_1` — no other values
- [ ] Write `test_C_selects_signal_index_not_window_position` — argmax(C) in {0,1,2,3,4}
- [ ] Write `test_C_index_range_0_to_4` — argmax(C[k]) always in {0,1,2,3,4}
- [ ] Write `test_y_shape_is_n_windows_by_1` — y has shape (N, 1), NOT (N, 5)
- [ ] Write `test_y_is_scalar_per_sample` — y[k] is a single value
- [ ] Write `test_y_matches_clean_value_of_C_selected_signal` — y[k] == clean[argmax(C[k])][k+W//2]
- [ ] Write `test_y_when_C_selects_s1_equals_s1_clean_at_center`
- [ ] Write `test_y_when_C_selects_s2_equals_s2_clean_at_center`
- [ ] Write `test_y_when_C_selects_s3_equals_s3_clean_at_center`
- [ ] Write `test_y_when_C_selects_s4_equals_s4_clean_at_center`
- [ ] Write `test_y_when_C_selects_s5_equals_s5_clean_at_center`
- [ ] Write `test_label_uses_center_index_k_plus_W_half`
- [ ] Write `test_build_windows_no_nan_in_X`
- [ ] Write `test_build_windows_no_nan_in_y`
- [ ] Write `test_build_windows_no_inf_in_X`
- [ ] Write `test_build_windows_no_inf_in_y`
- [ ] Write `test_reproducibility_same_seed_same_C`
- [ ] Write `test_different_seed_different_C`
- [ ] Write `test_build_windows_raises_on_signal_shorter_than_window`
- [ ] Write `test_all_signal_indices_appear_in_C_over_large_N` — uniform coverage over {0,1,2,3,4}
- [ ] Write `test_model_input_concat_shape_is_W_plus_5` — verify concat([x_w, C]) shape == (15,) for W=10
- [ ] Write `test_windower_rejects_source_other_than_s5_noisy` — raises ValueError if s5 key missing
- [ ] Write `test_y_shape_is_N_by_1_not_N` — y.ndim == 2 and y.shape[1] == 1 (not flat 1D)
- [ ] Write `test_center_index_uses_integer_division` — center == k + W // 2 (floor, not ceil or round)
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

---

## Phase 14 — `services/windower.py` — GREEN Phase

- [ ] Create `src/signal_dataset/services/windower.py`
- [ ] Add module docstring
- [ ] Import `numpy as np`, `constants`, config types
- [ ] Define `WindowResult` dataclass: `X`, `C`, `y`
- [ ] Add docstring to `WindowResult`
- [ ] Define `Windower` class
- [ ] Add class-level docstring
- [ ] Implement `__init__(self, window_size: int, n_signals: int, seed: int)`
- [ ] Store `self._W`, `self._n_signals = 5`, `self._rng`
- [ ] Implement `_generate_one_hot(self) -> np.ndarray`
  — length `n_signals` (=5), not W; random signal index
- [ ] Implement `build_windows(self, noisy_signals: dict, clean_signals: dict) -> WindowResult`
- [ ] Extract source signal: `src = noisy_signals["s5"]` — composite noisy ONLY
- [ ] Compute `N = len(src) - W + 1`
- [ ] Pre-allocate `X (N, W)`, `C (N, n_signals)`, `y (N, 1)` arrays
- [ ] Loop over k:
  - [ ] `X[k] = src[k : k+W]` — composite noisy window
  - [ ] `C[k] = _generate_one_hot()` — shape (5,), picks signal index i
  - [ ] `i = np.argmax(C[k])`
  - [ ] `signal_key = SIGNAL_IDS[i]` (e.g. "s2")
  - [ ] `y[k] = clean_signals[signal_key][k + W//2]` — scalar label
- [ ] Raise `ValueError` if source signal shorter than W
- [ ] Return `WindowResult(X=X, C=C, y=y)`
- [ ] Verify file is ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

---

## Phase 14 — `services/windower.py` — REFACTOR Phase

- [ ] Extract label-filling loop into `_build_labels` private method
- [ ] Extract window-slicing loop into `_build_features` private method
- [ ] Confirm `WindowResult` dataclass is the sole return type
- [ ] Run `uv run ruff check src/signal_dataset/services/windower.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 15 — `services/dataset_builder.py` — TDD RED Phase

- [ ] Create `tests/unit/test_dataset_builder.py`
- [ ] Add module docstring
- [ ] Write `test_dataset_builder_instantiates`
- [ ] Write `test_split_sizes_sum_to_total` — n_train + n_val + n_test == N
- [ ] Write `test_train_size_approx_70_percent`
- [ ] Write `test_val_size_approx_15_percent`
- [ ] Write `test_test_size_approx_15_percent`
- [ ] Write `test_split_is_chronological` — no shuffle, train comes first in time
- [ ] Write `test_save_dataset_creates_npz_file`
- [ ] Write `test_dataset_npz_has_X_train_key`
- [ ] Write `test_dataset_npz_has_X_val_key`
- [ ] Write `test_dataset_npz_has_X_test_key`
- [ ] Write `test_dataset_npz_has_C_train_key` — C arrays must be stored
- [ ] Write `test_dataset_npz_has_C_val_key`
- [ ] Write `test_dataset_npz_has_C_test_key`
- [ ] Write `test_dataset_npz_has_y_train_key`
- [ ] Write `test_dataset_npz_has_y_val_key`
- [ ] Write `test_dataset_npz_has_y_test_key`
- [ ] Write `test_X_train_shape_second_dim_equals_W`
- [ ] Write `test_X_val_shape_second_dim_equals_W`
- [ ] Write `test_X_test_shape_second_dim_equals_W`
- [ ] Write `test_C_train_shape_second_dim_equals_5` — NOT W
- [ ] Write `test_C_val_shape_second_dim_equals_5`
- [ ] Write `test_C_test_shape_second_dim_equals_5`
- [ ] Write `test_y_train_shape_second_dim_equals_1` — scalar label, NOT 5
- [ ] Write `test_y_val_shape_second_dim_equals_1`
- [ ] Write `test_y_test_shape_second_dim_equals_1`
- [ ] Write `test_C_train_rows_are_valid_one_hot` — each row sums to 1
- [ ] Write `test_y_train_values_match_selected_signal_clean_value`
- [ ] Write `test_save_raw_signals_creates_npz_file`
- [ ] Write `test_raw_npz_has_clean_signals_key`
- [ ] Write `test_raw_npz_has_noisy_signals_key`
- [ ] Write `test_raw_npz_has_time_axis_key`
- [ ] Write `test_clean_signals_shape_in_raw_npz` — (5, 10000)
- [ ] Write `test_noisy_signals_shape_in_raw_npz`
- [ ] Write `test_time_axis_shape_in_raw_npz` — (10000,)
- [ ] Write `test_no_nan_in_saved_X_train`
- [ ] Write `test_no_nan_in_saved_y_train`
- [ ] Write `test_raw_npz_contains_all_5_clean_signals` — s1_clean through s5_clean all present
- [ ] Write `test_raw_npz_contains_all_5_noisy_signals` — s1_noisy through s5_noisy all present
- [ ] Write `test_y_train_shape_is_2d` — y_train.ndim == 2, shape (N_train, 1) not (N_train,)
- [ ] Write `test_C_train_every_row_exactly_one_1` — no row has 0 or 2+ ones
- [ ] Write `test_split_train_ratio_from_config` — n_train / N ≈ cfg.train_ratio ± 1/N
- [ ] Write `test_split_val_ratio_from_config` — n_val / N ≈ cfg.val_ratio ± 1/N
- [ ] Write `test_split_test_ratio_from_config` — n_test / N ≈ cfg.test_ratio ± 1/N
- [ ] Write `test_split_chronological_no_overlap` — train ends where val begins, val ends where test begins
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

---

## Phase 15 — `services/dataset_builder.py` — GREEN Phase

- [ ] Create `src/signal_dataset/services/dataset_builder.py`
- [ ] Add module docstring
- [ ] Import `numpy as np`, `pathlib.Path`, `constants`, config types
- [ ] Define `SplitResult` dataclass: `X_train`, `C_train`, `y_train`, `X_val`, `C_val`, `y_val`, `X_test`, `C_test`, `y_test`
- [ ] Add docstring to `SplitResult`
- [ ] Define `DatasetBuilder` class
- [ ] Add class-level docstring
- [ ] Implement `__init__(self, cfg: DatasetConfig, output_cfg: OutputConfig)`
- [ ] Implement `split(self, X: np.ndarray, C: np.ndarray, y: np.ndarray) -> SplitResult`
- [ ] Compute indices: `n_train = int(N * train_ratio)`, `n_val = int(N * val_ratio)`
- [ ] Slice X, C, y chronologically — no shuffle — same indices for all three
- [ ] Return `SplitResult` with all 9 arrays (X, C, y per split)
- [ ] Implement `save_dataset(self, split: SplitResult) -> Path`
- [ ] Create parent directories if not existing
- [ ] Use `np.savez_compressed` — keys: `X_train`, `C_train`, `y_train`, `X_val`, `C_val`, `y_val`, `X_test`, `C_test`, `y_test`
- [ ] Return path to saved file
- [ ] Implement `save_raw_signals(self, clean: dict, noisy: dict, t: np.ndarray) -> Path`
- [ ] Stack clean signals into (5, N) array using `SIGNAL_IDS` order
- [ ] Stack noisy signals into (5, N) array
- [ ] Save with `np.savez_compressed` using keys from `constants`
- [ ] Verify file is ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

---

## Phase 15 — `services/dataset_builder.py` — REFACTOR Phase

- [ ] Confirm `SplitResult` dataclass replaces all tuple returns
- [ ] Extract directory creation into utility function in `shared` if used elsewhere
- [ ] Run `uv run ruff check src/signal_dataset/services/dataset_builder.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 16 — `sdk/sdk.py` — TDD RED Phase

- [ ] Create `tests/unit/test_sdk.py`
- [ ] Add module docstring
- [ ] Write `test_sdk_instantiates_with_config_path`
- [ ] Write `test_sdk_generate_dataset_returns_dataset_result`
- [ ] Write `test_dataset_result_has_dataset_path_attribute`
- [ ] Write `test_dataset_result_has_raw_path_attribute`
- [ ] Write `test_dataset_result_has_split_result_attribute`
- [ ] Write `test_dataset_result_dataset_path_is_path_object`
- [ ] Write `test_dataset_result_raw_path_is_path_object`
- [ ] Write `test_generate_dataset_creates_dataset_file` — file exists after call
- [ ] Write `test_generate_dataset_creates_raw_signals_file`
- [ ] Write `test_sdk_raises_config_error_on_bad_path`
- [ ] Write `test_sdk_raises_config_error_on_invalid_json`
- [ ] Write `test_sdk_get_version_returns_string`
- [ ] Write `test_sdk_get_version_starts_with_1`
- [ ] Write `test_sdk_generate_calls_signal_generator` (mock)
- [ ] Write `test_sdk_generate_calls_noise_injector` (mock)
- [ ] Write `test_sdk_generate_calls_windower` (mock)
- [ ] Write `test_sdk_generate_calls_dataset_builder` (mock)
- [ ] Verify test file ≤ 150 lines
- [ ] Run tests — confirm all FAIL (RED)

---

## Phase 16 — `sdk/sdk.py` — GREEN Phase

- [ ] Create `src/signal_dataset/sdk/sdk.py`
- [ ] Add module docstring explaining this is the single public API
- [ ] Import all service classes and shared modules
- [ ] Define `DatasetResult` dataclass: `dataset_path`, `raw_path`, `split_result`
- [ ] Add docstring to `DatasetResult`
- [ ] Define `DatasetSDK` class
- [ ] Add class-level docstring
- [ ] Implement `__init__(self, config_path: str | Path, rate_limits_path: str | Path)`
- [ ] Instantiate `ConfigManager` and call `load()`
- [ ] Instantiate `ApiGatekeeper` with loaded rate-limit config
- [ ] Store config as `self._cfg`
- [ ] Implement `generate_dataset(self) -> DatasetResult`
- [ ] Step 1: `SignalGenerator.generate_all()` → clean signals
- [ ] Step 2: `NoiseInjector.inject_all()` → noisy signals
- [ ] Step 3: `Windower.build_windows()` → WindowResult (X from s5_noisy, C one-hot length 5, y scalar)
- [ ] Step 4: `DatasetBuilder.split(X, C, y)` → SplitResult (9 arrays)
- [ ] Step 5: `DatasetBuilder.save_dataset()` → dataset path (includes C arrays)
- [ ] Step 6: `DatasetBuilder.save_raw_signals()` → raw path
- [ ] Return `DatasetResult`
- [ ] Implement `get_version(self) -> str` returning `__version__`
- [ ] Verify file is ≤ 150 lines
- [ ] Run tests — confirm all PASS (GREEN)

---

## Phase 16 — `sdk/sdk.py` — REFACTOR Phase

- [ ] Confirm no business logic leaks into SDK (orchestration only)
- [ ] Ensure each service is called exactly once in sequence
- [ ] Confirm `DatasetResult` carries all needed info for consumers
- [ ] Run `uv run ruff check src/signal_dataset/sdk/sdk.py` → 0 errors
- [ ] Re-run tests — still PASS

---

## Phase 17 — `main.py` CLI Entry Point

- [ ] Create `src/signal_dataset/main.py`
- [ ] Add module docstring
- [ ] Import `argparse`, `pathlib`, `sys`, `DatasetSDK`
- [ ] Define `parse_args() -> argparse.Namespace`
- [ ] Add `--config` argument (default: `config/setup.json`)
- [ ] Add `--rate-limits` argument (default: `config/rate_limits.json`)
- [ ] Add `--output-dir` argument (default: `data/`)
- [ ] Add `--verbose` flag
- [ ] Define `main() -> int`
- [ ] Parse args, instantiate `DatasetSDK`, call `generate_dataset()`
- [ ] Print summary: paths saved, shapes, time elapsed
- [ ] Return exit code 0 on success, 1 on error
- [ ] Add `if __name__ == "__main__": sys.exit(main())`
- [ ] Test: `uv run python -m signal_dataset --help` outputs usage
- [ ] Test: `uv run python -m signal_dataset` runs to completion
- [ ] Verify file is ≤ 150 lines
- [ ] Verify `uv run ruff check src/signal_dataset/main.py` → 0 errors

---

## Phase 18 — Integration Tests

- [ ] Create `tests/integration/test_pipeline.py`
- [ ] Add module docstring
- [ ] Write `test_full_pipeline_runs_without_error` — DatasetSDK end-to-end
- [ ] Write `test_full_pipeline_creates_dataset_npz` — file exists
- [ ] Write `test_full_pipeline_creates_raw_signals_npz` — file exists
- [ ] Write `test_dataset_npz_loadable` — np.load without error
- [ ] Write `test_raw_npz_loadable`
- [ ] Write `test_X_train_dtype_is_float` — float32 or float64
- [ ] Write `test_y_train_dtype_is_float`
- [ ] Write `test_C_train_dtype_is_float_or_int`
- [ ] Write `test_X_train_no_nan`
- [ ] Write `test_X_train_no_inf`
- [ ] Write `test_y_train_no_nan`
- [ ] Write `test_y_train_no_inf`
- [ ] Write `test_X_val_no_nan`
- [ ] Write `test_X_test_no_nan`
- [ ] Write `test_y_val_no_nan`
- [ ] Write `test_y_test_no_nan`
- [ ] Write `test_X_train_shape_second_dim_equals_W` — (N_train, 10)
- [ ] Write `test_C_train_shape_second_dim_equals_5` — (N_train, 5) NOT (N_train, 10)
- [ ] Write `test_y_train_shape_second_dim_equals_1` — (N_train, 1) NOT (N_train, 5)
- [ ] Write `test_C_train_rows_all_sum_to_1` — valid one-hot in all rows
- [ ] Write `test_X_train_windows_match_s5_noisy_slices` — verify source is composite
- [ ] Write `test_y_train_label_matches_selected_clean_signal` — y[k] == clean[argmax(C[k])][center]
- [ ] Write `test_train_val_test_sizes_sum_to_total_windows`
- [ ] Write `test_total_windows_equals_10000_minus_W_plus_1`
- [ ] Write `test_clean_signals_in_raw_npz_have_correct_frequencies` — FFT check
- [ ] Write `test_noisy_signals_differ_from_clean`
- [ ] Write `test_composite_noisy_in_raw_npz_equals_sum_of_noisy_components`
- [ ] Write `test_dataset_reproducible_with_same_seed` — run twice with seed=42, arrays are identical
- [ ] Write `test_model_input_concat_shape_is_15` — concat(X_train[0], C_train[0]) shape == (15,)
- [ ] Write `test_split_ratios_match_config_70_15_15` — n_train/N ≈ 0.70, n_val/N ≈ 0.15, n_test/N ≈ 0.15
- [ ] Write `test_signals_raw_contains_5_clean_and_5_noisy` — all 10 vectors present
- [ ] Write `test_config_change_propagates_to_output` — change window size, verify X second dim changes
- [ ] Verify test file ≤ 150 lines
- [ ] Run integration tests — confirm all PASS

---

## Phase 19 — Quality Gates

- [ ] Run `uv run ruff check src/` → 0 errors
- [ ] Run `uv run ruff check tests/` → 0 errors
- [ ] Run `uv run ruff check` on every individual file and fix any issues
- [ ] Run `uv run pytest tests/unit/` → all pass
- [ ] Run `uv run pytest tests/integration/` → all pass
- [ ] Run `uv run pytest tests/ --cov=src --cov-report=term-missing` → ≥ 85%
- [ ] Check `src/signal_dataset/__init__.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/constants.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/main.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/sdk/sdk.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/services/signal_generator.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/services/noise_injector.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/services/windower.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/services/dataset_builder.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/shared/config.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/shared/gatekeeper.py` ≤ 150 lines
- [ ] Check `src/signal_dataset/shared/version.py` ≤ 150 lines
- [ ] Check `tests/conftest.py` ≤ 150 lines
- [ ] Check `tests/unit/test_signal_generator.py` ≤ 150 lines
- [ ] Check `tests/unit/test_noise_injector.py` ≤ 150 lines
- [ ] Check `tests/unit/test_windower.py` ≤ 150 lines
- [ ] Check `tests/unit/test_dataset_builder.py` ≤ 150 lines
- [ ] Check `tests/unit/test_config.py` ≤ 150 lines
- [ ] Check `tests/unit/test_gatekeeper.py` ≤ 150 lines
- [ ] Check `tests/unit/test_sdk.py` ≤ 150 lines
- [ ] Check `tests/integration/test_pipeline.py` ≤ 150 lines
- [ ] Grep for any hardcoded `1000` in `.py` files — must not appear outside config loader
- [ ] Grep for any hardcoded `10000` in `.py` files — must not appear outside config loader
- [ ] Grep for any hardcoded `0.70`, `0.15` in `.py` files — must come from config
- [ ] Grep for any hardcoded `2.0`, `1.5`, `0.8`, `0.3` amplitude values in `.py` files
- [ ] Grep for any hardcoded `5`, `15`, `50`, `100` frequency values in `.py` files
- [ ] Verify `.env` not in git index
- [ ] Verify `uv.lock` exists and is up to date
- [ ] Verify `pyproject.toml` is the only dependency file

---

## Phase 20 — Visualisation: `results/clean_signals.png`

- [ ] Create visualisation script or notebook cell for clean signals plot
- [ ] Import `matplotlib.pyplot`, `numpy`, `DatasetSDK`
- [ ] Load or generate clean signals via SDK
- [ ] Create figure with 5 subplots (one per signal: s1, s2, s3, s4, s5)
- [ ] Plot s1 time-domain waveform — label "s1: 2.0·sin(2π·5·t)"
- [ ] Plot s2 time-domain waveform — label "s2: 1.5·sin(2π·15·t + π/4)"
- [ ] Plot s3 time-domain waveform — label "s3: 0.8·sin(2π·50·t)"
- [ ] Plot s4 time-domain waveform — label "s4: 0.3·sin(2π·100·t)"
- [ ] Plot s5 (composite) time-domain waveform
- [ ] Add x-axis label "Time (s)" to each subplot
- [ ] Add y-axis label "Amplitude" to each subplot
- [ ] Add title to each subplot
- [ ] Add grid to each subplot
- [ ] Limit x-axis to first 0.5 seconds for visibility (high-freq signals)
- [ ] Set figure title "Clean Sine Signal Components"
- [ ] Tight layout applied
- [ ] Save as `results/clean_signals.png` at 300 DPI
- [ ] Verify file saved and readable
- [ ] Verify file is not empty (size > 0 bytes)

---

## Phase 21 — Visualisation: `results/noisy_vs_clean.png`

- [ ] Create figure with 4 rows × 2 columns (one pair per base signal)
- [ ] For each of s1, s2, s3, s4:
- [ ] Left column: clean signal (first 0.5 sec)
- [ ] Right column: noisy signal (first 0.5 sec)
- [ ] Add "Clean" label to left column header
- [ ] Add "Noisy" label to right column header
- [ ] Use same y-axis scale for clean and noisy in each row
- [ ] Colour clean traces in blue, noisy traces in orange
- [ ] Add legend per subplot
- [ ] Add signal formula as subplot title
- [ ] Add grid to all subplots
- [ ] Set figure title "Clean vs Noisy Signals"
- [ ] Tight layout applied
- [ ] Save as `results/noisy_vs_clean.png` at 300 DPI
- [ ] Verify file saved correctly

---

## Phase 22 — Visualisation: `results/noise_distribution.png`

- [ ] Compute N_amp for all 4 signals
- [ ] Compute N_phase for all 4 signals
- [ ] Create figure with 2 rows × 4 columns (row=type, col=signal)
- [ ] Row 1: N_amp histograms per signal (s1–s4)
- [ ] Row 2: N_phase histograms per signal (s1–s4)
- [ ] Overlay fitted Gaussian PDF curve on each histogram
- [ ] Display measured σ and configured σ in each plot
- [ ] Add x-axis label "Noise Value"
- [ ] Add y-axis label "Density"
- [ ] Add subplot titles "N_amp s1", "N_amp s2", etc.
- [ ] Highlight burst-noise contribution separately (different colour bar)
- [ ] Set figure title "Noise Distribution Analysis"
- [ ] Tight layout applied
- [ ] Save as `results/noise_distribution.png` at 300 DPI
- [ ] Verify file saved correctly

---

## Phase 23 — Visualisation: `results/frequency_spectrum.png`

- [ ] Compute FFT of each clean signal (s1–s5)
- [ ] Compute FFT of each noisy signal (s1–s5)
- [ ] Create figure with 5 rows × 2 columns (clean FFT | noisy FFT)
- [ ] Plot magnitude spectrum (positive frequencies only) for each
- [ ] Add vertical dashed line at expected dominant frequency for s1–s4
- [ ] Add x-axis label "Frequency (Hz)"
- [ ] Add y-axis label "Magnitude"
- [ ] Set x-axis limit to [0, 200] Hz
- [ ] Use log scale on y-axis for better visibility
- [ ] Add subplot titles with signal ID and frequency
- [ ] Add grid to all subplots
- [ ] Set figure title "Frequency Spectrum: Clean vs Noisy"
- [ ] Tight layout applied
- [ ] Save as `results/frequency_spectrum.png` at 300 DPI
- [ ] Verify file saved correctly

---

## Phase 24 — Visualisation: `results/burst_events.png`

- [ ] Select signal s1 for burst demonstration
- [ ] Plot clean s1 (first 1.0 second)
- [ ] Plot noisy s1 on same axis
- [ ] Shade burst-event regions in red with alpha=0.3
- [ ] Add horizontal dashed lines at ±(A + burst_magnitude) to show expected extremes
- [ ] Add legend: "Clean", "Noisy", "Burst region"
- [ ] Add x-axis label "Time (s)"
- [ ] Add y-axis label "Amplitude"
- [ ] Add title "Burst Noise Events on s1"
- [ ] Add grid
- [ ] Create second subplot: N_amp over time with burst events highlighted
- [ ] Save as `results/burst_events.png` at 300 DPI
- [ ] Verify file saved correctly

---

## Phase 25 — Visualisation: `results/dataset_splits.png`

- [ ] Create bar chart showing train/val/test sample counts
- [ ] Add percentage labels above each bar
- [ ] Add x-axis labels: Train, Validation, Test
- [ ] Add y-axis label "Number of Windows"
- [ ] Add title "Dataset Split Distribution"
- [ ] Add grid on y-axis
- [ ] Add text box with total window count
- [ ] Save as `results/dataset_splits.png` at 300 DPI
- [ ] Verify file saved correctly

---

## Phase 26 — Visualisation: `results/context_window_example.png`

- [ ] Select 5 example windows from the dataset (one for each possible C index 0–4)
- [ ] For each window create 2 subplots side-by-side:
  - [ ] Left: line plot of x_w (composite noisy window, 10 samples)
  - [ ] Right: bar chart of C vector (5 bars, signal labels s1–s5 on x-axis)
- [ ] Highlight the selected bar (C[i]=1) in a distinct colour (e.g. red), rest grey
- [ ] Add text annotation on left plot: "Label: y = {value:.3f} ({signal_id}_clean)"
- [ ] Add x-axis label "Sample index within window" on left subplot
- [ ] Add x-axis label "Signal" on right subplot (s1, s2, s3, s4, s5)
- [ ] Add y-axis label "Amplitude" on left subplot
- [ ] Add y-axis label "C value (0 or 1)" on right subplot
- [ ] Add subplot row titles "Window k=... → extract s{i+1}"
- [ ] Set figure title "Context Window + One-Hot Signal Selector Examples"
- [ ] Tight layout applied
- [ ] Save as `results/context_window_example.png` at 300 DPI
- [ ] Verify file saved correctly
- [ ] Verify NO "x_feature" label appears anywhere in the plot (x_feature is obsolete)

---

## Phase 27 — Jupyter Notebook: `notebooks/dataset_analysis.ipynb`

- [ ] Create `notebooks/dataset_analysis.ipynb`
- [ ] Cell 1 — Title markdown: "# Signal Dataset Analysis"
- [ ] Cell 2 — Imports: numpy, matplotlib, seaborn, scipy, pathlib, sys
- [ ] Cell 3 — Set sys.path to include src/
- [ ] Cell 4 — Import DatasetSDK, instantiate, call generate_dataset()
- [ ] Cell 5 — Markdown: "## 1. Signal Overview"
- [ ] Cell 6 — Load clean signals from raw .npz
- [ ] Cell 7 — Plot all 5 clean signals (time domain, first 0.5 sec)
- [ ] Cell 8 — Print signal statistics: min, max, mean, std per signal
- [ ] Cell 9 — Markdown: "## 2. Noise Characterisation"
- [ ] Cell 10 — Compute N_amp per signal
- [ ] Cell 11 — Plot N_amp histograms with Gaussian fit overlaid
- [ ] Cell 12 — Compute and print: measured σ vs configured σ for each signal
- [ ] Cell 13 — Compute N_phase per signal
- [ ] Cell 14 — Plot N_phase histograms with Gaussian fit overlaid
- [ ] Cell 15 — Identify burst events: print count, mean duration, max duration
- [ ] Cell 16 — Markdown: "## 3. Frequency Analysis"
- [ ] Cell 17 — Compute FFT of clean signals
- [ ] Cell 18 — Plot FFT magnitude spectrum for all 5 clean signals
- [ ] Cell 19 — Verify dominant frequency peaks match specification (table)
- [ ] Cell 20 — Compute FFT of noisy signals
- [ ] Cell 21 — Plot FFT comparison: clean vs noisy for each base signal
- [ ] Cell 22 — Markdown: "## 4. Dataset Split Statistics"
- [ ] Cell 23 — Load dataset.npz, print shapes of all **9** arrays (X_train, C_train, y_train, X_val, C_val, y_val, X_test, C_test, y_test)
- [ ] Cell 24 — Print train/val/test sizes and percentages; verify they match config ratios (70/15/15)
- [ ] Cell 25 — Check for NaN/Inf in all 9 arrays — print PASS/FAIL per array
- [ ] Cell 26 — Plot distribution of y_train values; group by selected signal (argmax(C_train))
- [ ] Cell 27 — Markdown: "## 5. Context Window Visualisation"
- [ ] Cell 28 — Sample 10 windows from X_train; for each show x_w (line) and the C-selected signal label
- [ ] Cell 29 — Visualise C_train vectors as heatmap (**10 windows × 5 signal classes**); x-axis: s1–s5
- [ ] Cell 30 — Compute frequency of each signal index in C_train; verify approximately uniform over {0,1,2,3,4}
- [ ] Cell 31 — Markdown: "## 6. Summary & Conclusions"
- [ ] Cell 32 — Table: signal specs vs measured properties
- [ ] Cell 33 — Table: noise parameters configured vs measured
- [ ] Cell 34 — Table: dataset shape summary
- [ ] Cell 35 — Conclusions text on readiness for model training phase
- [ ] Run all cells — confirm no errors
- [ ] Verify all plots render inline
- [ ] Save notebook with output included (`jupyter nbconvert --to notebook --execute`)

---

## Phase 28 — README.md

- [ ] Create `README.md` at project root
- [ ] Add project title: "Noisy Sine Signal Dataset Generator"
- [ ] Add one-paragraph overview of the project purpose
- [ ] Add "## Signals" section with table of all 5 signals and formulas
- [ ] Add "## Noise Model" section explaining Gaussian + burst noise
- [ ] Add "## Dataset Structure" section explaining context window and labels
- [ ] Add "## Quickstart" section
- [ ] Add step: clone/download repo
- [ ] Add step: `uv sync`
- [ ] Add step: `uv run python -m signal_dataset`
- [ ] Add step: explain output files (`data/dataset.npz`, `data/signals_raw.npz`)
- [ ] Add "## Project Structure" section with annotated directory tree
- [ ] Add "## Configuration" section explaining `config/setup.json` fields
- [ ] Add "## Testing" section: `uv run pytest tests/`, coverage badge info
- [ ] Add "## Linting" section: `uv run ruff check`
- [ ] Add "## Results" section with embedded visualisation images
- [ ] Embed `results/clean_signals.png` in README
- [ ] Embed `results/noisy_vs_clean.png` in README
- [ ] Embed `results/frequency_spectrum.png` in README
- [ ] Add "## Next Phase" section mentioning RNN / FC / LSTM training
- [ ] Add "## License" section
- [ ] Add "## Author" section with contact email
- [ ] Verify README renders correctly as markdown
- [ ] Verify all image paths are correct relative to repo root

---

## Phase 29 — `docs/PRD_dataset_generation.md`

- [ ] Create `docs/PRD_dataset_generation.md`
- [ ] Add title and version header
- [ ] Add "Algorithm: Dataset Generation Pipeline" section
- [ ] Document INPUT DATA: config paths, signal parameters, noise parameters
- [ ] Document OUTPUT DATA: dataset.npz keys and shapes, signals_raw.npz keys and shapes
- [ ] Document SETUP DATA: all configurable parameters with defaults and valid ranges
- [ ] Write out the full noise injection algorithm step-by-step
- [ ] Write out the full windowing algorithm step-by-step
- [ ] Write out the full chronological split algorithm
- [ ] Add pseudocode for the complete pipeline
- [ ] Add parameter sensitivity analysis plan (OAT): vary sigma_amp, sigma_phase, window_size, burst_probability
- [ ] List expected effects of each parameter change
- [ ] Document known edge cases and how they are handled
- [ ] Add references / citations for Gaussian white noise and burst noise models
- [ ] Submit for approval

---

## Phase 30 — Prompt Engineering Log

- [ ] Create `docs/prompt_engineering_log.md`
- [ ] Add header: "# Prompt Engineering Log"
- [ ] Document Prompt 1: initial project description provided to AI
- [ ] Document Prompt 2: PRD/PLAN/TODO generation
- [ ] Document AI model used: claude-sonnet-4-6
- [ ] Document date of each prompt
- [ ] Document outcome of each prompt (what was generated)
- [ ] Document any iterative refinements made
- [ ] Document any corrections applied to AI output
- [ ] Keep log updated as development proceeds

---

## Phase 31 — Final Submission Checklist

- [ ] All docs present: PRD.md, PLAN.md, TODO.md, PRD_dataset_generation.md, README.md
- [ ] Prompt Engineering Log complete
- [ ] `uv run ruff check` → 0 errors across entire project
- [ ] `uv run pytest tests/ --cov=src` → ≥ 85% coverage
- [ ] All source files ≤ 150 lines
- [ ] All test files ≤ 150 lines
- [ ] No hardcoded signal parameters in Python source
- [ ] No API keys or secrets in source
- [ ] `.env-example` committed with dummy values
- [ ] `.env` in `.gitignore` and not tracked
- [ ] `uv.lock` committed and up to date
- [ ] `pyproject.toml` is sole dependency declaration
- [ ] No `requirements.txt` present
- [ ] No `pip install` used anywhere
- [ ] `data/dataset.npz` generated and loadable
- [ ] `data/signals_raw.npz` generated and loadable — contains all 5 clean + 5 noisy vectors
- [ ] All **7** visualisation `.png` files present in `results/`: clean_signals, noisy_vs_clean, noise_distribution, frequency_spectrum, burst_events, dataset_splits, context_window_example
- [ ] dataset.npz contains all 9 arrays: X_train, C_train, y_train, X_val, C_val, y_val, X_test, C_test, y_test
- [ ] C arrays in dataset.npz are valid one-hot (every row sums to 1, second dim = 5)
- [ ] y arrays in dataset.npz have shape (N_split, 1) — scalar per sample, not multi-output
- [ ] X arrays sourced from s5_noisy only — verified by comparing to raw npz
- [ ] Running generator twice with seed 42 produces byte-identical dataset.npz (reproducibility)
- [ ] `notebooks/dataset_analysis.ipynb` executed end-to-end without errors (`jupyter nbconvert --execute`)
- [ ] All notebook cells have output saved
- [ ] `DatasetSDK` is sole entry point confirmed (no service imported directly in main.py)
- [ ] `ApiGatekeeper` wired through all future external calls
- [ ] `ConfigManager` used for all config reads
- [ ] `version.py` reflects correct version string "1.00"
- [ ] Git history has structured commit messages
- [ ] Each logical unit committed separately (scaffold, config, services, tests, docs)
- [ ] README.md renders with correct images
- [ ] Integration test passes on clean environment (`uv sync` → `uv run pytest`)
- [ ] Model training phase (RNN/FC/LSTM) confirmed as out of scope for this submission
- [ ] Supervisor final review completed
- [ ] Submission packaged and delivered

---

## Blocked / Notes

- Model training (RNN / FC / LSTM) is out of scope for this phase
- Supervisor approval required before Phase 1 coding begins
- `data/` and `results/` are git-ignored; regenerate locally with `uv run python -m signal_dataset`
- Burst noise implementation must use `np.random.default_rng` (not legacy `np.random`) for reproducibility
