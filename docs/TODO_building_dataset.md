# TODO — Dataset Generation Phase



**Version:** 1.00

**Date:** 2026-05-10

**Status:** Complete



Legend: `[ ]` = pending · `[~]` = in progress · `[x]` = done



---



## Phase 0 — Documentation & Approval



- [x] Read PRD.md in full and verify signal definitions match specification

- [x] Verify s1 formula: 2.0 · sin(2π · 5 · t) in PRD.md

- [x] Verify s2 formula: 1.5 · sin(2π · 15 · t + π/4) in PRD.md

- [x] Verify s3 formula: 0.8 · sin(2π · 50 · t) in PRD.md

- [x] Verify s4 formula: 0.3 · sin(2π · 100 · t) in PRD.md

- [x] Verify s5 = s1 + s2 + s3 + s4 stated clearly in PRD.md

- [x] Verify noise model equation in PRD.md: `(A + N_amp) · sin(2πft + φ + N_phase)`

- [x] Verify Gaussian white noise term documented in PRD.md

- [x] Verify burst noise term documented in PRD.md

- [x] Verify context window size W=10 documented in PRD.md

- [x] Verify one-hot vector C definition in PRD.md

- [x] Verify C is one-hot of length **5** (signal selector, NOT element-wise mask on window samples) in PRD.md

- [x] Verify model-input concatenation documented in PRD.md: `x_input = concat([x_w, C])` → shape `(W+5,)` = `(15,)`

- [x] Verify label definition (clean signal at window center) in PRD.md

- [x] Verify train/val/test split ratios (70/15/15) in PRD.md

- [x] Verify config schema documented in PRD.md

- [x] Verify acceptance criteria complete in PRD.md

- [x] Read PLAN.md in full and verify module responsibilities

- [x] Verify DatasetSDK as single entry point in PLAN.md

- [x] Verify all 5 services listed in PLAN.md

- [x] Verify ApiGatekeeper in PLAN.md

- [x] Verify data flow diagram correct in PLAN.md

- [x] Verify noise model math in PLAN.md matches PRD.md

- [x] Verify file size budget all ≤ 150 lines in PLAN.md

- [x] Verify testing strategy lists all test files in PLAN.md

- [x] Verify 5 visualisations planned in PLAN.md

- [x] Read TODO.md (this file) and verify all phases present

- [ ] PRD.md submitted to supervisor for approval

- [ ] PLAN.md submitted to supervisor for approval

- [ ] TODO.md submitted to supervisor for approval

- [x] PRD_dataset_generation.md drafted (per-algorithm PRD)

- [ ] PRD_dataset_generation.md submitted for approval

- [ ] Supervisor approval received on PRD.md

- [ ] Supervisor approval received on PLAN.md

- [ ] Supervisor approval received on PRD_dataset_generation.md

- [ ] Supervisor approval received on TODO.md

- [ ] All approvals recorded with date and version



---



## Phase 1 — Environment Setup



- [x] Verify Python 3.13 is installed (`python --version`)

- [x] Verify uv is installed (`uv --version`)

- [x] Verify ruff is available (`ruff --version`)

- [x] Run `uv init signal_dataset` to initialise project

- [x] Confirm `pyproject.toml` created by uv init

- [x] Confirm `uv.lock` created after init

- [x] Run `uv add numpy` and verify added to `pyproject.toml`

- [x] Run `uv add matplotlib` and verify added

- [x] Run `uv add seaborn` and verify added

- [x] Run `uv add plotly` and verify added

- [x] Run `uv add jupyter` and verify added

- [x] Run `uv add scipy` and verify added (for FFT / statistical tests)

- [x] Run `uv add --dev pytest` and verify added to dev deps

- [x] Run `uv add --dev pytest-cov` and verify added

- [x] Run `uv add --dev ruff` and verify added

- [x] Run `uv add --dev pytest-mock` and verify added

- [x] Run `uv lock` and verify `uv.lock` updated

- [x] Run `uv sync` and verify all packages install without errors

- [x] Confirm no `requirements.txt` exists (forbidden)

- [x] Confirm no `pip install` commands used anywhere



---



## Phase 2 — Folder Structure Creation



- [x] Create `src/` directory

- [x] Create `src/signal_dataset/` directory

- [x] Create `src/signal_dataset/sdk/` directory

- [x] Create `src/signal_dataset/services/` directory

- [x] Create `src/signal_dataset/shared/` directory

- [x] Create `tests/` directory

- [x] Create `tests/unit/` directory

- [x] Create `tests/integration/` directory

- [x] Create `config/` directory

- [x] Create `data/` directory

- [x] Create `results/` directory

- [x] Create `assets/` directory

- [x] Create `notebooks/` directory

- [x] Create `docs/` directory (already exists)

- [x] Verify full tree matches PLAN.md exactly



---



## Phase 3 — pyproject.toml Configuration



- [x] Set `[project] name = "signal-dataset"`

- [x] Set `[project] version = "1.0.0"`

- [x] Set `[project] requires-python = ">=3.10"`

- [x] Add `numpy` to `[project] dependencies`

- [x] Add `matplotlib` to `[project] dependencies`

- [x] Add `seaborn` to `[project] dependencies`

- [x] Add `plotly` to `[project] dependencies`

- [x] Add `scipy` to `[project] dependencies`

- [x] Add `jupyter` to `[project] dependencies`

- [x] Add `pytest` to `[project.optional-dependencies] dev`

- [x] Add `pytest-cov` to dev deps

- [x] Add `ruff` to dev deps

- [x] Add `pytest-mock` to dev deps

- [x] Add `[tool.ruff]` section with `line-length = 100`

- [x] Add `[tool.ruff]` section with `target-version = "py310"`

- [x] Add `[tool.ruff.lint] select = ["E","F","W","I","N","UP","B","C4","SIM"]`

- [x] Add `[tool.ruff.lint] ignore = ["E501"]`

- [x] Add `[tool.pytest.ini_options] testpaths = ["tests"]`

- [x] Add `[tool.pytest.ini_options] addopts = "--cov=src --cov-report=term-missing"`

- [x] Add `[tool.coverage.run] source = ["src"]`

- [x] Add `[tool.coverage.run] omit = ["src/main.py", "*/tests/*"]`

- [x] Add `[tool.coverage.report] fail_under = 85`

- [x] Add `[build-system]` table with hatchling or setuptools

- [x] Add `[project.scripts] signal-dataset = "signal_dataset.main:main"`

- [x] Verify `uv run ruff check pyproject.toml` passes

- [x] Verify `uv sync` still works after all edits



---



## Phase 4 — Configuration Files



- [x] Create `config/setup.json`

- [x] Add `"dataset"` block to `setup.json`

- [x] Add `"duration_sec": 10` to dataset block

- [x] Add `"sample_rate_hz": 1000` to dataset block

- [x] Add `"window_size": 10` to dataset block

- [x] Add `"train_ratio": 0.70` to dataset block

- [x] Add `"val_ratio": 0.15` to dataset block

- [x] Add `"test_ratio": 0.15` to dataset block

- [x] Add `"random_seed": 42` to dataset block

- [x] Add `"signals"` array to `setup.json`

- [x] Add s1 entry: `{"id":"s1","amplitude":2.0,"frequency_hz":5,"phase_rad":0}`

- [x] Add s2 entry: `{"id":"s2","amplitude":1.5,"frequency_hz":15,"phase_rad":0.7854}`

- [x] Add s3 entry: `{"id":"s3","amplitude":0.8,"frequency_hz":50,"phase_rad":0}`

- [x] Add s4 entry: `{"id":"s4","amplitude":0.3,"frequency_hz":100,"phase_rad":0}`

- [x] Add `"noise"` block to `setup.json`

- [x] Add `"gaussian"` sub-block under noise

- [x] Add `"sigma_amp": [0.05, 0.05, 0.03, 0.02]` to gaussian block

- [x] Add `"sigma_phase": [0.05, 0.05, 0.03, 0.02]` to gaussian block

- [x] Add `"burst"` sub-block under noise

- [x] Add `"probability": 0.01` to burst block

- [x] Add `"duration_range_samples": [5, 20]` to burst block

- [x] Add `"amp_magnitude": 0.5` to burst block

- [x] Add `"phase_magnitude": 0.3` to burst block

- [x] Add `"output"` block to `setup.json`

- [x] Add `"dataset_path": "data/dataset.npz"` to output block

- [x] Add `"raw_signals_path": "data/signals_raw.npz"` to output block

- [x] Add `"results_dir": "results/"` to output block

- [x] Validate `setup.json` is valid JSON (`python -m json.tool config/setup.json`)

- [x] Create `config/rate_limits.json`

- [x] Add `"rate_limits"` top-level key

- [x] Add `"version": "1.00"` inside rate_limits

- [x] Add `"services"` block inside rate_limits

- [x] Add `"default"` service with `"requests_per_minute": 30`

- [x] Add `"requests_per_hour": 500` to default service

- [x] Add `"concurrent_max": 5` to default service

- [x] Add `"retry_after_seconds": 30` to default service

- [x] Add `"max_retries": 3` to default service

- [x] Add `"queue_max_depth": 100` to default service

- [x] Validate `rate_limits.json` is valid JSON

- [x] Confirm no numeric literals appear in any `.py` file that should come from config



---



## Phase 5 — .gitignore and .env-example



- [x] Create `.gitignore`

- [x] Add `.env` to `.gitignore`

- [x] Add `*.key` to `.gitignore`

- [x] Add `*.pem` to `.gitignore`

- [x] Add `credentials.json` to `.gitignore`

- [x] Add `data/` to `.gitignore`

- [x] Add `results/` to `.gitignore`

- [x] Add `__pycache__/` to `.gitignore`

- [x] Add `*.pyc` to `.gitignore`

- [x] Add `.pytest_cache/` to `.gitignore`

- [x] Add `.coverage` to `.gitignore`

- [x] Add `htmlcov/` to `.gitignore`

- [x] Add `.ruff_cache/` to `.gitignore`

- [x] Add `*.egg-info/` to `.gitignore`

- [x] Add `.venv/` to `.gitignore`

- [x] Add `dist/` to `.gitignore`

- [x] Add `build/` to `.gitignore`

- [x] Add `*.ipynb_checkpoints` to `.gitignore`

- [x] Create `.env-example`

- [x] Add `API_KEY=your_api_key_here` (dummy) to `.env-example`

- [x] Add `LOG_LEVEL=INFO` to `.env-example`

- [x] Add `DATA_DIR=data/` to `.env-example`

- [x] Add `RESULTS_DIR=results/` to `.env-example`

- [x] Verify `.env` is not committed

- [x] Verify `.env-example` is committed



---



## Phase 6 — `src/signal_dataset/shared/version.py`



- [x] Create file `src/signal_dataset/shared/version.py`

- [x] Add module docstring explaining version tracking purpose

- [x] Define `__version__ = "1.00"`

- [x] Define `VERSION_MAJOR = 1`

- [x] Define `VERSION_MINOR = 0`

- [x] Define `BUILD_DATE = "2026-05-10"`

- [x] Define `get_version()` function returning `__version__`

- [x] Add docstring to `get_version()` explaining return value

- [x] Verify file is ≤ 150 lines

- [x] Verify `uv run ruff check src/signal_dataset/shared/version.py` → 0 errors



---



## Phase 7 — `src/signal_dataset/constants.py`



- [x] Create file `src/signal_dataset/constants.py`

- [x] Add module docstring explaining this is the single source for all physical/math constants

- [x] Define `TWO_PI = 2.0 * math.pi`

- [x] Define `DEFAULT_DURATION_SEC = 10` (fallback only, config takes priority)

- [x] Define `DEFAULT_SAMPLE_RATE_HZ = 1000`

- [x] Define `DEFAULT_WINDOW_SIZE = 10`

- [x] Define `N_SIGNALS = 4` (number of base sine components)

- [x] Define `N_TOTAL_SIGNALS = 5` (including composite)

- [x] Define `COMPOSITE_SIGNAL_ID = "s5"`

- [x] Define `SIGNAL_IDS = ["s1", "s2", "s3", "s4", "s5"]`

- [x] Define `DATASET_NPZ_KEY_X_TRAIN = "X_train"`

- [x] Define `DATASET_NPZ_KEY_X_VAL = "X_val"`

- [x] Define `DATASET_NPZ_KEY_X_TEST = "X_test"`

- [x] Define `DATASET_NPZ_KEY_C_TRAIN = "C_train"` — one-hot signal selector

- [x] Define `DATASET_NPZ_KEY_C_VAL = "C_val"`

- [x] Define `DATASET_NPZ_KEY_C_TEST = "C_test"`

- [x] Define `DATASET_NPZ_KEY_Y_TRAIN = "y_train"`

- [x] Define `DATASET_NPZ_KEY_Y_VAL = "y_val"`

- [x] Define `DATASET_NPZ_KEY_Y_TEST = "y_test"`

- [x] Define `N_SELECTOR_CLASSES = 5` — length of C vector (one per signal)

- [x] Define `RAW_NPZ_KEY_CLEAN = "clean_signals"`

- [x] Define `RAW_NPZ_KEY_NOISY = "noisy_signals"`

- [x] Define `RAW_NPZ_KEY_TIME = "time_axis"`

- [x] Define `NOISE_TYPE_GAUSSIAN = "gaussian"`

- [x] Define `NOISE_TYPE_BURST = "burst"`

- [x] Import only `math` (no numpy in constants)

- [x] Verify file is ≤ 150 lines

- [x] Verify `uv run ruff check src/signal_dataset/constants.py` → 0 errors

- [x] Verify no business logic in this file — constants only



---



## Phase 8 — `src/signal_dataset/__init__.py`



- [x] Create `src/signal_dataset/__init__.py`

- [x] Add module-level docstring describing the package

- [x] Import `__version__` from `shared.version`

- [x] Define `__all__` list: `["DatasetSDK", "__version__"]`

- [x] Import `DatasetSDK` from `sdk.sdk`

- [x] Verify file is ≤ 150 lines

- [x] Verify `uv run ruff check src/signal_dataset/__init__.py` → 0 errors

- [x] Create `src/signal_dataset/sdk/__init__.py` (empty, or minimal)

- [x] Create `src/signal_dataset/services/__init__.py` (empty, or minimal)

- [x] Create `src/signal_dataset/shared/__init__.py` (empty, or minimal)



---



## Phase 9 — `tests/conftest.py` (Shared Fixtures)



- [x] Create `tests/conftest.py`

- [x] Add module docstring explaining shared fixtures

- [x] Import `pytest`, `numpy`, `pathlib.Path`

- [x] Define `config_path` fixture returning path to `config/setup.json`

- [x] Define `rate_limits_path` fixture returning path to `config/rate_limits.json`

- [x] Define `sample_config_dict` fixture with minimal valid config dict (no file I/O)

- [x] Define `time_axis_fixture` fixture: `np.linspace(0, 10, 10000, endpoint=False)`

- [x] Define `small_time_axis` fixture: 100-sample axis for fast tests

- [x] Define `clean_s1_fixture` — precomputed s1 for small axis

- [x] Define `clean_s2_fixture` — precomputed s2 for small axis

- [x] Define `clean_s3_fixture` — precomputed s3 for small axis

- [x] Define `clean_s4_fixture` — precomputed s4 for small axis

- [x] Define `clean_s5_fixture` — sum of the four above

- [x] Define `noisy_signals_fixture` — dict with 5 noisy arrays (small axis)

- [x] Define `tmp_data_dir` fixture using `tmp_path` for isolated file I/O

- [x] Define `window_size_fixture` returning 10

- [x] Define `random_seed_fixture` returning 42

- [x] Create `tests/unit/__init__.py` (empty)

- [x] Create `tests/integration/__init__.py` (empty)

- [x] Verify conftest.py is ≤ 150 lines

- [x] Verify `uv run ruff check tests/conftest.py` → 0 errors



---



## Phase 10 — `shared/config.py` — TDD RED Phase



- [x] Create `tests/unit/test_config.py`

- [x] Add module docstring to test file

- [x] Write `test_config_manager_loads_valid_json` — ConfigManager loads setup.json without error

- [x] Write `test_config_manager_returns_dataset_section` — `cfg.dataset` not None

- [x] Write `test_config_duration_sec_equals_10` — `cfg.dataset.duration_sec == 10`

- [x] Write `test_config_sample_rate_hz_equals_1000` — `cfg.dataset.sample_rate_hz == 1000`

- [x] Write `test_config_window_size_equals_10` — `cfg.dataset.window_size == 10`

- [x] Write `test_config_train_ratio_equals_0_70` — `cfg.dataset.train_ratio == 0.70`

- [x] Write `test_config_val_ratio_equals_0_15`

- [x] Write `test_config_test_ratio_equals_0_15`

- [x] Write `test_config_split_ratios_sum_to_1` — train+val+test == 1.0

- [x] Write `test_config_random_seed_equals_42`

- [x] Write `test_config_signals_list_has_4_entries` — len(cfg.signals) == 4

- [x] Write `test_config_s1_amplitude_equals_2_0`

- [x] Write `test_config_s1_frequency_equals_5`

- [x] Write `test_config_s1_phase_equals_0`

- [x] Write `test_config_s2_amplitude_equals_1_5`

- [x] Write `test_config_s2_frequency_equals_15`

- [x] Write `test_config_s2_phase_approx_pi_over_4`

- [x] Write `test_config_s3_amplitude_equals_0_8`

- [x] Write `test_config_s3_frequency_equals_50`

- [x] Write `test_config_s3_phase_equals_0`

- [x] Write `test_config_s4_amplitude_equals_0_3`

- [x] Write `test_config_s4_frequency_equals_100`

- [x] Write `test_config_s4_phase_equals_0`

- [x] Write `test_config_s2_phase_equals_0_7854` — exact value 0.7854 rad (π/4)

- [x] Write `test_config_noise_gaussian_sigma_amp_has_4_entries`

- [x] Write `test_config_noise_gaussian_sigma_amp_values_exact` — [0.05, 0.05, 0.03, 0.02]

- [x] Write `test_config_noise_gaussian_sigma_amp_s1_equals_0_05`

- [x] Write `test_config_noise_gaussian_sigma_amp_s2_equals_0_05`

- [x] Write `test_config_noise_gaussian_sigma_amp_s3_equals_0_03`

- [x] Write `test_config_noise_gaussian_sigma_amp_s4_equals_0_02`

- [x] Write `test_config_noise_gaussian_sigma_phase_has_4_entries`

- [x] Write `test_config_noise_gaussian_sigma_phase_values_exact` — [0.05, 0.05, 0.03, 0.02]

- [x] Write `test_config_noise_gaussian_sigma_phase_s1_equals_0_05`

- [x] Write `test_config_noise_gaussian_sigma_phase_s2_equals_0_05`

- [x] Write `test_config_noise_gaussian_sigma_phase_s3_equals_0_03`

- [x] Write `test_config_noise_gaussian_sigma_phase_s4_equals_0_02`

- [x] Write `test_config_noise_burst_probability_equals_0_01` — exact value, not just positive

- [x] Write `test_config_noise_burst_duration_min_equals_5` — exact lower bound

- [x] Write `test_config_noise_burst_duration_max_equals_20` — exact upper bound

- [x] Write `test_config_noise_burst_duration_range_valid` — min < max, both positive

- [x] Write `test_config_noise_burst_amp_magnitude_equals_0_5` — exact value

- [x] Write `test_config_noise_burst_phase_magnitude_equals_0_3` — exact value

- [x] Write `test_config_raises_on_missing_file` — FileNotFoundError on bad path

- [x] Write `test_config_raises_on_invalid_json` — JSONDecodeError or ConfigValidationError

- [x] Write `test_config_raises_on_negative_duration`

- [x] Write `test_config_raises_on_zero_sample_rate`

- [x] Write `test_config_raises_on_window_size_zero`

- [x] Write `test_config_raises_on_split_ratios_not_summing_to_1`

- [x] Write `test_config_raises_on_negative_amplitude`

- [x] Write `test_config_raises_on_negative_frequency`

- [x] Write `test_config_raises_on_empty_signals_list`

- [x] Write `test_config_raises_on_negative_noise_sigma`

- [x] Write `test_config_raises_on_burst_probability_above_1`

- [x] Write `test_config_loads_rate_limits_json`

- [x] Write `test_config_rate_limit_default_service_exists`

- [x] Write `test_config_rate_limit_requests_per_minute_positive`

- [x] Verify test file is ≤ 150 lines (split into `test_config_signals.py` if needed)

- [x] Run tests — confirm all FAIL (RED state confirmed)



---



## Phase 10 — `shared/config.py` — GREEN Phase



- [x] Create `src/signal_dataset/shared/config.py`

- [x] Add module docstring explaining config loading and validation

- [x] Import `json`, `os`, `pathlib`, `dataclasses`, `typing`

- [x] Define `SignalConfig` dataclass: `id`, `amplitude`, `frequency_hz`, `phase_rad`

- [x] Add docstring to `SignalConfig`

- [x] Define `GaussianNoiseConfig` dataclass: `sigma_amp`, `sigma_phase`

- [x] Add docstring to `GaussianNoiseConfig`

- [x] Define `BurstNoiseConfig` dataclass: `probability`, `duration_range_samples`, `amp_magnitude`, `phase_magnitude`

- [x] Add docstring to `BurstNoiseConfig`

- [x] Define `NoiseConfig` dataclass: `gaussian`, `burst`

- [x] Add docstring to `NoiseConfig`

- [x] Define `DatasetConfig` dataclass: `duration_sec`, `sample_rate_hz`, `window_size`, `train_ratio`, `val_ratio`, `test_ratio`, `random_seed`

- [x] Add docstring to `DatasetConfig`

- [x] Define `OutputConfig` dataclass: `dataset_path`, `raw_signals_path`, `results_dir`

- [x] Add docstring to `OutputConfig`

- [x] Define `AppConfig` dataclass: `dataset`, `signals`, `noise`, `output`

- [x] Add docstring to `AppConfig`

- [x] Define `ConfigValidationError(Exception)` custom exception

- [x] Define `ConfigManager` class

- [x] Add class-level docstring to `ConfigManager`

- [x] Implement `ConfigManager.__init__(self, config_path, rate_limits_path)`

- [x] Implement `ConfigManager.load() -> AppConfig`

- [x] Implement private `_load_json(path) -> dict` with error handling

- [x] Implement private `_parse_dataset(raw) -> DatasetConfig`

- [x] Implement private `_parse_signals(raw) -> list[SignalConfig]`

- [x] Implement private `_parse_noise(raw) -> NoiseConfig`

- [x] Implement private `_parse_output(raw) -> OutputConfig`

- [x] Implement private `_validate(config: AppConfig)` raising `ConfigValidationError`

- [x] Validate `duration_sec > 0`

- [x] Validate `sample_rate_hz > 0`

- [x] Validate `window_size > 0`

- [x] Validate `train_ratio + val_ratio + test_ratio ≈ 1.0` (tolerance 1e-9)

- [x] Validate all ratios in (0, 1)

- [x] Validate all signal amplitudes > 0

- [x] Validate all signal frequencies > 0

- [x] Validate sigma_amp all ≥ 0

- [x] Validate sigma_phase all ≥ 0

- [x] Validate burst probability in [0, 1]

- [x] Validate burst duration_range_samples[0] < duration_range_samples[1]

- [x] Validate len(sigma_amp) == len(signals)

- [x] Verify file is ≤ 150 lines (split into `config_models.py` for dataclasses if needed)

- [x] Run tests — confirm all PASS (GREEN state confirmed)



---



## Phase 10 — `shared/config.py` — REFACTOR Phase



- [x] Check for any duplicated parsing logic — extract helpers

- [x] Ensure each method is single-responsibility

- [x] Ensure all error messages are descriptive (include field name and bad value)

- [x] Remove any dead code or commented-out lines

- [x] Confirm ≤ 150 lines total

- [x] Run `uv run ruff check src/signal_dataset/shared/config.py` → 0 errors

- [x] Re-run tests — still PASS after refactor



---



## Phase 11 — `shared/gatekeeper.py` — TDD RED Phase



- [x] Create `tests/unit/test_gatekeeper.py`

- [x] Add module docstring to test file

- [x] Write `test_gatekeeper_instantiates_with_valid_config`

- [x] Write `test_gatekeeper_execute_calls_function`

- [x] Write `test_gatekeeper_execute_returns_function_result`

- [x] Write `test_gatekeeper_execute_logs_call` — check call is recorded in log

- [x] Write `test_gatekeeper_execute_single_callable_no_error`

- [x] Write `test_gatekeeper_queue_status_returns_queue_status_object`

- [x] Write `test_gatekeeper_queue_status_depth_starts_at_0`

- [x] Write `test_gatekeeper_queue_status_processed_starts_at_0`

- [x] Write `test_gatekeeper_queue_status_failed_starts_at_0`

- [x] Write `test_gatekeeper_get_call_log_returns_list`

- [x] Write `test_gatekeeper_get_call_log_is_empty_initially`

- [x] Write `test_gatekeeper_get_call_log_grows_after_execute`

- [x] Write `test_gatekeeper_call_log_entry_has_timestamp`

- [x] Write `test_gatekeeper_call_log_entry_has_function_name`

- [x] Write `test_gatekeeper_call_log_entry_has_status_success`

- [x] Write `test_gatekeeper_call_log_entry_has_status_failure_on_exception`

- [x] Write `test_gatekeeper_execute_retries_on_exception`

- [x] Write `test_gatekeeper_execute_raises_after_max_retries`

- [x] Write `test_gatekeeper_execute_respects_max_retries_config`

- [x] Write `test_gatekeeper_queue_depth_not_exceeded` — backpressure logic

- [x] Write `test_gatekeeper_raises_when_queue_full`

- [x] Verify test file is ≤ 150 lines

- [x] Run tests — confirm all FAIL (RED)



---



## Phase 11 — `shared/gatekeeper.py` — GREEN Phase



- [x] Create `src/signal_dataset/shared/gatekeeper.py`

- [x] Add module docstring explaining centralised API call management

- [x] Import `time`, `queue`, `threading`, `dataclasses`, `typing`, `logging`

- [x] Define `RateLimitConfig` dataclass: `requests_per_minute`, `requests_per_hour`, `concurrent_max`, `retry_after_seconds`, `max_retries`, `queue_max_depth`

- [x] Add docstring to `RateLimitConfig`

- [x] Define `QueueStatus` dataclass: `depth`, `processed`, `failed`

- [x] Add docstring to `QueueStatus`

- [x] Define `CallLogEntry` dataclass: `timestamp`, `function_name`, `status`, `error_message`

- [x] Add docstring to `CallLogEntry`

- [x] Define `ApiGatekeeper` class

- [x] Add class-level docstring

- [x] Implement `__init__(self, config: RateLimitConfig)`

- [x] Initialise FIFO queue (`queue.Queue`) with maxsize from config

- [x] Initialise `_call_log: list[CallLogEntry]`

- [x] Initialise counters: `_processed`, `_failed`

- [x] Implement `execute(self, api_call, *args, **kwargs)` with retry loop

- [x] Implement retry logic: up to `max_retries`, sleep `retry_after_seconds` between

- [x] Log each attempt to `_call_log`

- [x] Raise `GatekeeperQueueFullError` when queue at max depth

- [x] Implement `get_queue_status(self) -> QueueStatus`

- [x] Implement `get_call_log(self) -> list[CallLogEntry]`

- [x] Define `GatekeeperQueueFullError(Exception)`

- [x] Define `GatekeeperMaxRetriesError(Exception)`

- [x] Verify file is ≤ 150 lines

- [x] Run tests — confirm all PASS (GREEN)



---



## Phase 11 — `shared/gatekeeper.py` — REFACTOR Phase



- [x] Extract retry logic into private `_execute_with_retry` method

- [x] Extract log-entry creation into private `_log_call` method

- [x] Ensure thread safety — review lock usage

- [x] Confirm all docstrings present

- [x] Run `uv run ruff check src/signal_dataset/shared/gatekeeper.py` → 0 errors

- [x] Re-run tests — still PASS



---



## Phase 12 — `services/signal_generator.py` — TDD RED Phase



- [x] Create `tests/unit/test_signal_generator.py`

- [x] Add module docstring

- [x] Write `test_signal_generator_instantiates`

- [x] Write `test_generate_time_axis_shape` — shape == (10000,)

- [x] Write `test_generate_time_axis_start_equals_0` — t[0] == 0.0

- [x] Write `test_generate_time_axis_last_sample_equals_9_999` — t[-1] == 9.999 exactly (endpoint=False)

- [x] Write `test_generate_time_axis_end_less_than_duration` — t[-1] < 10.0

- [x] Write `test_generate_time_axis_step_equals_1_over_sample_rate` — t[1]-t[0] == 0.001

- [x] Write `test_generate_s1_shape` — (10000,)

- [x] Write `test_generate_s2_shape`

- [x] Write `test_generate_s3_shape`

- [x] Write `test_generate_s4_shape`

- [x] Write `test_generate_s5_shape`

- [x] Write `test_generate_s1_max_amplitude_approx_2_0` — np.max(abs(s1)) ≈ 2.0

- [x] Write `test_generate_s2_max_amplitude_approx_1_5`

- [x] Write `test_generate_s3_max_amplitude_approx_0_8`

- [x] Write `test_generate_s4_max_amplitude_approx_0_3`

- [x] Write `test_generate_s1_is_zero_mean` — np.mean(s1) ≈ 0 (tolerance 0.01)

- [x] Write `test_generate_s2_is_zero_mean`

- [x] Write `test_generate_s3_is_zero_mean`

- [x] Write `test_generate_s4_is_zero_mean`

- [x] Write `test_generate_s5_equals_sum_of_components` — np.allclose(s5, s1+s2+s3+s4)

- [x] Write `test_generate_s1_dominant_frequency_via_fft` — FFT peak at 5 Hz

- [x] Write `test_generate_s2_dominant_frequency_via_fft` — FFT peak at 15 Hz

- [x] Write `test_generate_s3_dominant_frequency_via_fft` — FFT peak at 50 Hz

- [x] Write `test_generate_s4_dominant_frequency_via_fft` — FFT peak at 100 Hz

- [x] Write `test_generate_s2_phase_offset` — verify π/4 phase via correlation

- [x] Write `test_generate_all_returns_dict_with_5_keys`

- [x] Write `test_generate_all_keys_are_s1_through_s5`

- [x] Write `test_generate_all_values_are_ndarrays`

- [x] Write `test_generate_no_nans_in_any_signal`

- [x] Write `test_generate_no_infs_in_any_signal`

- [x] Verify test file ≤ 150 lines

- [x] Run tests — confirm all FAIL (RED)



---



## Phase 12 — `services/signal_generator.py` — GREEN Phase



- [x] Create `src/signal_dataset/services/signal_generator.py`

- [x] Add module docstring

- [x] Import `numpy as np` and relevant constants from `constants.py`

- [x] Import `SignalConfig`, `DatasetConfig` from `shared.config`

- [x] Define `SignalGenerator` class

- [x] Add class-level docstring

- [x] Implement `__init__(self, dataset_cfg: DatasetConfig, signals: list[SignalConfig])`

- [x] Store `self._dataset_cfg` and `self._signals`

- [x] Implement `generate_time_axis(self) -> np.ndarray`

- [x] Use `np.linspace(0, duration, n_samples, endpoint=False)` — no hardcoded values

- [x] Implement `generate_single(self, signal_cfg: SignalConfig, t: np.ndarray) -> np.ndarray`

- [x] Formula: `A * np.sin(TWO_PI * f * t + phi)` using `TWO_PI` from constants

- [x] Implement `generate_composite(self, components: dict[str, np.ndarray]) -> np.ndarray`

- [x] Sum all component arrays (s1+s2+s3+s4)

- [x] Implement `generate_all(self) -> dict[str, np.ndarray]`

- [x] Call `generate_time_axis`, loop over signals, call `generate_single`, then `generate_composite`

- [x] Return dict with keys `"s1"` through `"s5"` using `SIGNAL_IDS` from constants

- [x] Verify file is ≤ 150 lines

- [x] Run tests — confirm all PASS (GREEN)



---



## Phase 12 — `services/signal_generator.py` — REFACTOR Phase



- [x] Ensure no magic numbers (`2.0`, `1.5`, etc.) remain in source — all from config

- [x] Check for code duplication across generate methods — extract if needed

- [x] Confirm every method has a docstring

- [x] Run `uv run ruff check src/signal_dataset/services/signal_generator.py` → 0 errors

- [x] Re-run tests — still PASS



---



## Phase 13 — `services/noise_injector.py` — TDD RED Phase



- [x] Create `tests/unit/test_noise_injector.py`

- [x] Add module docstring

- [x] Write `test_noise_injector_instantiates`

- [x] Write `test_generate_gaussian_amp_noise_shape` — shape == (10000,)

- [x] Write `test_generate_gaussian_amp_noise_mean_near_zero` — abs(mean) < 3*sigma/sqrt(N)

- [x] Write `test_generate_gaussian_amp_noise_std_near_sigma` — std ≈ sigma (tolerance 10%)

- [x] Write `test_generate_gaussian_phase_noise_shape`

- [x] Write `test_generate_gaussian_phase_noise_mean_near_zero`

- [x] Write `test_generate_gaussian_phase_noise_std_near_sigma`

- [x] Write `test_generate_burst_mask_is_binary` — all values in {0, 1}

- [x] Write `test_generate_burst_mask_shape` — shape == (10000,)

- [x] Write `test_generate_burst_mask_burst_durations_in_range` — check run-length of 1s

- [x] Write `test_generate_burst_noise_amp_shape`

- [x] Write `test_generate_burst_noise_phase_shape`

- [x] Write `test_inject_single_differs_from_clean` — not allclose

- [x] Write `test_inject_single_shape_preserved` — shape == clean signal shape

- [x] Write `test_inject_single_no_nan`

- [x] Write `test_inject_single_no_inf`

- [x] Write `test_inject_all_returns_dict_with_5_keys`

- [x] Write `test_inject_all_keys_match_signal_ids`

- [x] Write `test_inject_all_values_are_ndarrays`

- [x] Write `test_n_amp_and_n_phase_are_independent` — correlation between them near 0

- [x] Write `test_reproducibility_with_same_seed` — same seed → same noisy signal

- [x] Write `test_different_seeds_give_different_noise`

- [x] Write `test_zero_sigma_gaussian_produces_no_gaussian_noise` — noisy == clean when sigma=0 and no burst

- [x] Write `test_zero_burst_probability_produces_no_burst_events`

- [x] Write `test_burst_magnitude_bounded` — burst values ≤ amp_magnitude

- [x] Write `test_composite_noisy_differs_from_clean_composite`

- [x] Write `test_inject_all_s5_noisy_equals_sum_of_s1_s2_s3_s4_noisy` — unit test: s5_noisy == s1_noisy + s2_noisy + s3_noisy + s4_noisy (allclose)

- [x] Write `test_noise_formula_n_amp_is_additive_to_amplitude` — with constant N_amp offset and zero N_phase, verify peak amplitude shifts by offset

- [x] Write `test_noise_formula_n_phase_is_additive_to_phase` — with constant N_phase offset and zero N_amp, verify phase shift in output

- [x] Verify test file ≤ 150 lines (split if needed)

- [x] Run tests — confirm all FAIL (RED)



---



## Phase 13 — `services/noise_injector.py` — GREEN Phase



- [x] Create `src/signal_dataset/services/noise_injector.py`

- [x] Add module docstring

- [x] Import `numpy as np`, `constants`, `config` types

- [x] Define `NoiseInjector` class

- [x] Add class-level docstring

- [x] Implement `__init__(self, noise_cfg: NoiseConfig, dataset_cfg: DatasetConfig, seed: int)`

- [x] Store `self._rng = np.random.default_rng(seed)`

- [x] Implement `_generate_gaussian_noise(self, sigma: float, n: int) -> np.ndarray`

- [x] Implement `_generate_burst_mask(self, n: int) -> np.ndarray`

- [x] Use Bernoulli trial per sample with burst probability; extend runs to duration

- [x] Implement `_generate_burst_noise(self, mask: np.ndarray, magnitude: float) -> np.ndarray`

- [x] Uniform random values in [-magnitude, +magnitude] where mask is 1

- [x] Implement `_compute_n_amp(self, signal_idx: int, n: int) -> np.ndarray`

- [x] Sum gaussian_amp + burst_amp

- [x] Implement `_compute_n_phase(self, signal_idx: int, n: int) -> np.ndarray`

- [x] Sum gaussian_phase + burst_phase

- [x] Implement `inject_single(self, clean: np.ndarray, signal_cfg: SignalConfig, signal_idx: int) -> np.ndarray`

- [x] Formula: `(A + N_amp) * np.sin(TWO_PI * f * t + phi + N_phase)` — read t from stored axis

- [x] Implement `inject_all(self, clean_signals: dict, signal_configs: list, t: np.ndarray) -> dict`

- [x] Loop over s1-s4, call `inject_single`, compute s5_noisy as sum of noisy components

- [x] Verify file is ≤ 150 lines

- [x] Run tests — confirm all PASS (GREEN)



---



## Phase 13 — `services/noise_injector.py` — REFACTOR Phase



- [x] Extract burst run-length extension into `_expand_burst_mask` helper

- [x] Ensure `_generate_burst_mask` and `_generate_burst_noise` are independently testable

- [x] Confirm no shared mutable state between inject calls

- [x] Run `uv run ruff check src/signal_dataset/services/noise_injector.py` → 0 errors

- [x] Re-run tests — still PASS



---



## Phase 14 — `services/windower.py` — TDD RED Phase



- [x] Create `tests/unit/test_windower.py`

- [x] Add module docstring

- [x] Write `test_windower_instantiates`

- [x] Write `test_build_windows_X_shape_is_n_windows_by_W` — X has shape (N, W)

- [x] Write `test_n_windows_equals_n_samples_minus_W_plus_1`

- [x] Write `test_X_windows_sourced_from_s5_noisy_only` — X[k] == s5_noisy[k:k+W]

- [x] Write `test_X_does_not_equal_s1_noisy_window` — confirm s1_noisy is NOT the source

- [x] Write `test_X_does_not_equal_s2_noisy_window`

- [x] Write `test_X_does_not_equal_s3_noisy_window`

- [x] Write `test_X_does_not_equal_s4_noisy_window`

- [x] Write `test_C_shape_is_n_windows_by_5` — C has shape (N, 5), NOT (N, W)

- [x] Write `test_each_C_vector_is_one_hot` — np.sum(C[k]) == 1 for all k

- [x] Write `test_each_C_vector_has_exactly_one_nonzero`

- [x] Write `test_C_values_are_0_or_1` — no other values

- [x] Write `test_C_selects_signal_index_not_window_position` — argmax(C) in {0,1,2,3,4}

- [x] Write `test_C_index_range_0_to_4` — argmax(C[k]) always in {0,1,2,3,4}

- [x] Write `test_y_shape_is_n_windows_by_1` — y has shape (N, 1), NOT (N, 5)

- [x] Write `test_y_is_scalar_per_sample` — y[k] is a single value

- [x] Write `test_y_matches_clean_value_of_C_selected_signal` — y[k] == clean[argmax(C[k])][k+W//2]

- [x] Write `test_y_when_C_selects_s1_equals_s1_clean_at_center`

- [x] Write `test_y_when_C_selects_s2_equals_s2_clean_at_center`

- [x] Write `test_y_when_C_selects_s3_equals_s3_clean_at_center`

- [x] Write `test_y_when_C_selects_s4_equals_s4_clean_at_center`

- [x] Write `test_y_when_C_selects_s5_equals_s5_clean_at_center`

- [x] Write `test_label_uses_center_index_k_plus_W_half`

- [x] Write `test_build_windows_no_nan_in_X`

- [x] Write `test_build_windows_no_nan_in_y`

- [x] Write `test_build_windows_no_inf_in_X`

- [x] Write `test_build_windows_no_inf_in_y`

- [x] Write `test_reproducibility_same_seed_same_C`

- [x] Write `test_different_seed_different_C`

- [x] Write `test_build_windows_raises_on_signal_shorter_than_window`

- [x] Write `test_all_signal_indices_appear_in_C_over_large_N` — uniform coverage over {0,1,2,3,4}

- [x] Write `test_model_input_concat_shape_is_W_plus_5` — verify concat([x_w, C]) shape == (15,) for W=10

- [x] Write `test_windower_rejects_source_other_than_s5_noisy` — raises ValueError if s5 key missing

- [x] Write `test_y_shape_is_N_by_1_not_N` — y.ndim == 2 and y.shape[1] == 1 (not flat 1D)

- [x] Write `test_center_index_uses_integer_division` — center == k + W // 2 (floor, not ceil or round)

- [x] Verify test file ≤ 150 lines

- [x] Run tests — confirm all FAIL (RED)



---



## Phase 14 — `services/windower.py` — GREEN Phase



- [x] Create `src/signal_dataset/services/windower.py`

- [x] Add module docstring

- [x] Import `numpy as np`, `constants`, config types

- [x] Define `WindowResult` dataclass: `X`, `C`, `y`

- [x] Add docstring to `WindowResult`

- [x] Define `Windower` class

- [x] Add class-level docstring

- [x] Implement `__init__(self, window_size: int, n_signals: int, seed: int)`

- [x] Store `self._W`, `self._n_signals = 5`, `self._rng`

- [x] Implement `_generate_one_hot(self) -> np.ndarray`

  — length `n_signals` (=5), not W; random signal index

- [x] Implement `build_windows(self, noisy_signals: dict, clean_signals: dict) -> WindowResult`

- [x] Extract source signal: `src = noisy_signals["s5"]` — composite noisy ONLY

- [x] Compute `N = len(src) - W + 1`

- [x] Pre-allocate `X (N, W)`, `C (N, n_signals)`, `y (N, 1)` arrays

- [x] Loop over k:

  - [x] `X[k] = src[k : k+W]` — composite noisy window

  - [x] `C[k] = _generate_one_hot()` — shape (5,), picks signal index i

  - [x] `i = np.argmax(C[k])`

  - [x] `signal_key = SIGNAL_IDS[i]` (e.g. "s2")

  - [x] `y[k] = clean_signals[signal_key][k + W//2]` — scalar label

- [x] Raise `ValueError` if source signal shorter than W

- [x] Return `WindowResult(X=X, C=C, y=y)`

- [x] Verify file is ≤ 150 lines

- [x] Run tests — confirm all PASS (GREEN)



---



## Phase 14 — `services/windower.py` — REFACTOR Phase



- [x] Extract label-filling loop into `_build_labels` private method

- [x] Extract window-slicing loop into `_build_features` private method

- [x] Confirm `WindowResult` dataclass is the sole return type

- [x] Run `uv run ruff check src/signal_dataset/services/windower.py` → 0 errors

- [x] Re-run tests — still PASS



---



## Phase 15 — `services/dataset_builder.py` — TDD RED Phase



- [x] Create `tests/unit/test_dataset_builder.py`

- [x] Add module docstring

- [x] Write `test_dataset_builder_instantiates`

- [x] Write `test_split_sizes_sum_to_total` — n_train + n_val + n_test == N

- [x] Write `test_train_size_approx_70_percent`

- [x] Write `test_val_size_approx_15_percent`

- [x] Write `test_test_size_approx_15_percent`

- [x] Write `test_split_is_chronological` — no shuffle, train comes first in time

- [x] Write `test_save_dataset_creates_npz_file`

- [x] Write `test_dataset_npz_has_X_train_key`

- [x] Write `test_dataset_npz_has_X_val_key`

- [x] Write `test_dataset_npz_has_X_test_key`

- [x] Write `test_dataset_npz_has_C_train_key` — C arrays must be stored

- [x] Write `test_dataset_npz_has_C_val_key`

- [x] Write `test_dataset_npz_has_C_test_key`

- [x] Write `test_dataset_npz_has_y_train_key`

- [x] Write `test_dataset_npz_has_y_val_key`

- [x] Write `test_dataset_npz_has_y_test_key`

- [x] Write `test_X_train_shape_second_dim_equals_W`

- [x] Write `test_X_val_shape_second_dim_equals_W`

- [x] Write `test_X_test_shape_second_dim_equals_W`

- [x] Write `test_C_train_shape_second_dim_equals_5` — NOT W

- [x] Write `test_C_val_shape_second_dim_equals_5`

- [x] Write `test_C_test_shape_second_dim_equals_5`

- [x] Write `test_y_train_shape_second_dim_equals_1` — scalar label, NOT 5

- [x] Write `test_y_val_shape_second_dim_equals_1`

- [x] Write `test_y_test_shape_second_dim_equals_1`

- [x] Write `test_C_train_rows_are_valid_one_hot` — each row sums to 1

- [x] Write `test_y_train_values_match_selected_signal_clean_value`

- [x] Write `test_save_raw_signals_creates_npz_file`

- [x] Write `test_raw_npz_has_clean_signals_key`

- [x] Write `test_raw_npz_has_noisy_signals_key`

- [x] Write `test_raw_npz_has_time_axis_key`

- [x] Write `test_clean_signals_shape_in_raw_npz` — (5, 10000)

- [x] Write `test_noisy_signals_shape_in_raw_npz`

- [x] Write `test_time_axis_shape_in_raw_npz` — (10000,)

- [x] Write `test_no_nan_in_saved_X_train`

- [x] Write `test_no_nan_in_saved_y_train`

- [x] Write `test_raw_npz_contains_all_5_clean_signals` — s1_clean through s5_clean all present

- [x] Write `test_raw_npz_contains_all_5_noisy_signals` — s1_noisy through s5_noisy all present

- [x] Write `test_y_train_shape_is_2d` — y_train.ndim == 2, shape (N_train, 1) not (N_train,)

- [x] Write `test_C_train_every_row_exactly_one_1` — no row has 0 or 2+ ones

- [x] Write `test_split_train_ratio_from_config` — n_train / N ≈ cfg.train_ratio ± 1/N

- [x] Write `test_split_val_ratio_from_config` — n_val / N ≈ cfg.val_ratio ± 1/N

- [x] Write `test_split_test_ratio_from_config` — n_test / N ≈ cfg.test_ratio ± 1/N

- [x] Write `test_split_chronological_no_overlap` — train ends where val begins, val ends where test begins

- [x] Verify test file ≤ 150 lines

- [x] Run tests — confirm all FAIL (RED)



---



## Phase 15 — `services/dataset_builder.py` — GREEN Phase



- [x] Create `src/signal_dataset/services/dataset_builder.py`

- [x] Add module docstring

- [x] Import `numpy as np`, `pathlib.Path`, `constants`, config types

- [x] Define `SplitResult` dataclass: `X_train`, `C_train`, `y_train`, `X_val`, `C_val`, `y_val`, `X_test`, `C_test`, `y_test`

- [x] Add docstring to `SplitResult`

- [x] Define `DatasetBuilder` class

- [x] Add class-level docstring

- [x] Implement `__init__(self, cfg: DatasetConfig, output_cfg: OutputConfig)`

- [x] Implement `split(self, X: np.ndarray, C: np.ndarray, y: np.ndarray) -> SplitResult`

- [x] Compute indices: `n_train = int(N * train_ratio)`, `n_val = int(N * val_ratio)`

- [x] Slice X, C, y chronologically — no shuffle — same indices for all three

- [x] Return `SplitResult` with all 9 arrays (X, C, y per split)

- [x] Implement `save_dataset(self, split: SplitResult) -> Path`

- [x] Create parent directories if not existing

- [x] Use `np.savez_compressed` — keys: `X_train`, `C_train`, `y_train`, `X_val`, `C_val`, `y_val`, `X_test`, `C_test`, `y_test`

- [x] Return path to saved file

- [x] Implement `save_raw_signals(self, clean: dict, noisy: dict, t: np.ndarray) -> Path`

- [x] Stack clean signals into (5, N) array using `SIGNAL_IDS` order

- [x] Stack noisy signals into (5, N) array

- [x] Save with `np.savez_compressed` using keys from `constants`

- [x] Verify file is ≤ 150 lines

- [x] Run tests — confirm all PASS (GREEN)



---



## Phase 15 — `services/dataset_builder.py` — REFACTOR Phase



- [x] Confirm `SplitResult` dataclass replaces all tuple returns

- [x] Extract directory creation into utility function in `shared` if used elsewhere

- [x] Run `uv run ruff check src/signal_dataset/services/dataset_builder.py` → 0 errors

- [x] Re-run tests — still PASS



---



## Phase 16 — `sdk/sdk.py` — TDD RED Phase



- [x] Create `tests/unit/test_sdk.py`

- [x] Add module docstring

- [x] Write `test_sdk_instantiates_with_config_path`

- [x] Write `test_sdk_generate_dataset_returns_dataset_result`

- [x] Write `test_dataset_result_has_dataset_path_attribute`

- [x] Write `test_dataset_result_has_raw_path_attribute`

- [x] Write `test_dataset_result_has_split_result_attribute`

- [x] Write `test_dataset_result_dataset_path_is_path_object`

- [x] Write `test_dataset_result_raw_path_is_path_object`

- [x] Write `test_generate_dataset_creates_dataset_file` — file exists after call

- [x] Write `test_generate_dataset_creates_raw_signals_file`

- [x] Write `test_sdk_raises_config_error_on_bad_path`

- [x] Write `test_sdk_raises_config_error_on_invalid_json`

- [x] Write `test_sdk_get_version_returns_string`

- [x] Write `test_sdk_get_version_starts_with_1`

- [x] Write `test_sdk_generate_calls_signal_generator` (mock)

- [x] Write `test_sdk_generate_calls_noise_injector` (mock)

- [x] Write `test_sdk_generate_calls_windower` (mock)

- [x] Write `test_sdk_generate_calls_dataset_builder` (mock)

- [x] Verify test file ≤ 150 lines

- [x] Run tests — confirm all FAIL (RED)



---



## Phase 16 — `sdk/sdk.py` — GREEN Phase



- [x] Create `src/signal_dataset/sdk/sdk.py`

- [x] Add module docstring explaining this is the single public API

- [x] Import all service classes and shared modules

- [x] Define `DatasetResult` dataclass: `dataset_path`, `raw_path`, `split_result`

- [x] Add docstring to `DatasetResult`

- [x] Define `DatasetSDK` class

- [x] Add class-level docstring

- [x] Implement `__init__(self, config_path: str | Path, rate_limits_path: str | Path)`

- [x] Instantiate `ConfigManager` and call `load()`

- [x] Instantiate `ApiGatekeeper` with loaded rate-limit config

- [x] Store config as `self._cfg`

- [x] Implement `generate_dataset(self) -> DatasetResult`

- [x] Step 1: `SignalGenerator.generate_all()` → clean signals

- [x] Step 2: `NoiseInjector.inject_all()` → noisy signals

- [x] Step 3: `Windower.build_windows()` → WindowResult (X from s5_noisy, C one-hot length 5, y scalar)

- [x] Step 4: `DatasetBuilder.split(X, C, y)` → SplitResult (9 arrays)

- [x] Step 5: `DatasetBuilder.save_dataset()` → dataset path (includes C arrays)

- [x] Step 6: `DatasetBuilder.save_raw_signals()` → raw path

- [x] Return `DatasetResult`

- [x] Implement `get_version(self) -> str` returning `__version__`

- [x] Verify file is ≤ 150 lines

- [x] Run tests — confirm all PASS (GREEN)



---



## Phase 16 — `sdk/sdk.py` — REFACTOR Phase



- [x] Confirm no business logic leaks into SDK (orchestration only)

- [x] Ensure each service is called exactly once in sequence

- [x] Confirm `DatasetResult` carries all needed info for consumers

- [x] Run `uv run ruff check src/signal_dataset/sdk/sdk.py` → 0 errors

- [x] Re-run tests — still PASS



---



## Phase 17 — `main.py` CLI Entry Point



- [x] Create `src/signal_dataset/main.py`

- [x] Add module docstring

- [x] Import `argparse`, `pathlib`, `sys`, `DatasetSDK`

- [x] Define `parse_args() -> argparse.Namespace`

- [x] Add `--config` argument (default: `config/setup.json`)

- [x] Add `--rate-limits` argument (default: `config/rate_limits.json`)

- [x] Add `--output-dir` argument (default: `data/`)

- [x] Add `--verbose` flag

- [x] Define `main() -> int`

- [x] Parse args, instantiate `DatasetSDK`, call `generate_dataset()`

- [x] Print summary: paths saved, shapes, time elapsed

- [x] Return exit code 0 on success, 1 on error

- [x] Add `if __name__ == "__main__": sys.exit(main())`

- [x] Test: `uv run python -m signal_dataset --help` outputs usage

- [x] Test: `uv run python -m signal_dataset` runs to completion

- [x] Verify file is ≤ 150 lines

- [x] Verify `uv run ruff check src/signal_dataset/main.py` → 0 errors



---



## Phase 18 — Integration Tests



- [x] Create `tests/integration/test_pipeline.py`

- [x] Add module docstring

- [x] Write `test_full_pipeline_runs_without_error` — DatasetSDK end-to-end

- [x] Write `test_full_pipeline_creates_dataset_npz` — file exists

- [x] Write `test_full_pipeline_creates_raw_signals_npz` — file exists

- [x] Write `test_dataset_npz_loadable` — np.load without error

- [x] Write `test_raw_npz_loadable`

- [x] Write `test_X_train_dtype_is_float` — float32 or float64

- [x] Write `test_y_train_dtype_is_float`

- [x] Write `test_C_train_dtype_is_float_or_int`

- [x] Write `test_X_train_no_nan`

- [x] Write `test_X_train_no_inf`

- [x] Write `test_y_train_no_nan`

- [x] Write `test_y_train_no_inf`

- [x] Write `test_X_val_no_nan`

- [x] Write `test_X_test_no_nan`

- [x] Write `test_y_val_no_nan`

- [x] Write `test_y_test_no_nan`

- [x] Write `test_X_train_shape_second_dim_equals_W` — (N_train, 10)

- [x] Write `test_C_train_shape_second_dim_equals_5` — (N_train, 5) NOT (N_train, 10)

- [x] Write `test_y_train_shape_second_dim_equals_1` — (N_train, 1) NOT (N_train, 5)

- [x] Write `test_C_train_rows_all_sum_to_1` — valid one-hot in all rows

- [x] Write `test_X_train_windows_match_s5_noisy_slices` — verify source is composite

- [x] Write `test_y_train_label_matches_selected_clean_signal` — y[k] == clean[argmax(C[k])][center]

- [x] Write `test_train_val_test_sizes_sum_to_total_windows`

- [x] Write `test_total_windows_equals_10000_minus_W_plus_1`

- [x] Write `test_clean_signals_in_raw_npz_have_correct_frequencies` — FFT check

- [x] Write `test_noisy_signals_differ_from_clean`

- [x] Write `test_composite_noisy_in_raw_npz_equals_sum_of_noisy_components`

- [x] Write `test_dataset_reproducible_with_same_seed` — run twice with seed=42, arrays are identical

- [x] Write `test_model_input_concat_shape_is_15` — concat(X_train[0], C_train[0]) shape == (15,)

- [x] Write `test_split_ratios_match_config_70_15_15` — n_train/N ≈ 0.70, n_val/N ≈ 0.15, n_test/N ≈ 0.15

- [x] Write `test_signals_raw_contains_5_clean_and_5_noisy` — all 10 vectors present

- [x] Write `test_config_change_propagates_to_output` — change window size, verify X second dim changes

- [x] Verify test file ≤ 150 lines

- [x] Run integration tests — confirm all PASS



---



## Phase 19 — Quality Gates



- [x] Run `uv run ruff check src/` → 0 errors

- [x] Run `uv run ruff check tests/` → 0 errors

- [x] Run `uv run ruff check` on every individual file and fix any issues

- [x] Run `uv run pytest tests/unit/` → all pass

- [x] Run `uv run pytest tests/integration/` → all pass

- [x] Run `uv run pytest tests/ --cov=src --cov-report=term-missing` → ≥ 85%

- [x] Check `src/signal_dataset/__init__.py` ≤ 150 lines

- [x] Check `src/signal_dataset/constants.py` ≤ 150 lines

- [x] Check `src/signal_dataset/main.py` ≤ 150 lines

- [x] Check `src/signal_dataset/sdk/sdk.py` ≤ 150 lines

- [x] Check `src/signal_dataset/services/signal_generator.py` ≤ 150 lines

- [x] Check `src/signal_dataset/services/noise_injector.py` ≤ 150 lines

- [x] Check `src/signal_dataset/services/windower.py` ≤ 150 lines

- [x] Check `src/signal_dataset/services/dataset_builder.py` ≤ 150 lines

- [x] Check `src/signal_dataset/shared/config.py` ≤ 150 lines

- [x] Check `src/signal_dataset/shared/gatekeeper.py` ≤ 150 lines

- [x] Check `src/signal_dataset/shared/version.py` ≤ 150 lines

- [x] Check `tests/conftest.py` ≤ 150 lines

- [x] Check `tests/unit/test_signal_generator.py` ≤ 150 lines

- [x] Check `tests/unit/test_noise_injector.py` ≤ 150 lines

- [x] Check `tests/unit/test_windower.py` ≤ 150 lines

- [x] Check `tests/unit/test_dataset_builder.py` ≤ 150 lines

- [x] Check `tests/unit/test_config.py` ≤ 150 lines

- [x] Check `tests/unit/test_gatekeeper.py` ≤ 150 lines

- [x] Check `tests/unit/test_sdk.py` ≤ 150 lines

- [x] Check `tests/integration/test_pipeline.py` ≤ 150 lines

- [x] Grep for any hardcoded `1000` in `.py` files — must not appear outside config loader

- [x] Grep for any hardcoded `10000` in `.py` files — must not appear outside config loader

- [x] Grep for any hardcoded `0.70`, `0.15` in `.py` files — must come from config

- [x] Grep for any hardcoded `2.0`, `1.5`, `0.8`, `0.3` amplitude values in `.py` files

- [x] Grep for any hardcoded `5`, `15`, `50`, `100` frequency values in `.py` files

- [x] Verify `.env` not in git index

- [x] Verify `uv.lock` exists and is up to date

- [x] Verify `pyproject.toml` is the only dependency file



---



## Phase 20 — Visualisation: `results/clean_signals.png`



- [x] Create visualisation script or notebook cell for clean signals plot

- [x] Import `matplotlib.pyplot`, `numpy`, `DatasetSDK`

- [x] Load or generate clean signals via SDK

- [x] Create figure with 5 subplots (one per signal: s1, s2, s3, s4, s5)

- [x] Plot s1 time-domain waveform — label "s1: 2.0·sin(2π·5·t)"

- [x] Plot s2 time-domain waveform — label "s2: 1.5·sin(2π·15·t + π/4)"

- [x] Plot s3 time-domain waveform — label "s3: 0.8·sin(2π·50·t)"

- [x] Plot s4 time-domain waveform — label "s4: 0.3·sin(2π·100·t)"

- [x] Plot s5 (composite) time-domain waveform

- [x] Add x-axis label "Time (s)" to each subplot

- [x] Add y-axis label "Amplitude" to each subplot

- [x] Add title to each subplot

- [x] Add grid to each subplot

- [x] Limit x-axis to first 0.5 seconds for visibility (high-freq signals)

- [x] Set figure title "Clean Sine Signal Components"

- [x] Tight layout applied

- [x] Save as `results/clean_signals.png` at 300 DPI

- [x] Verify file saved and readable

- [x] Verify file is not empty (size > 0 bytes)



---



## Phase 21 — Visualisation: `results/noisy_vs_clean.png`



- [x] Create figure with 4 rows × 2 columns (one pair per base signal)

- [x] For each of s1, s2, s3, s4:

- [x] Left column: clean signal (first 0.5 sec)

- [x] Right column: noisy signal (first 0.5 sec)

- [x] Add "Clean" label to left column header

- [x] Add "Noisy" label to right column header

- [x] Use same y-axis scale for clean and noisy in each row

- [x] Colour clean traces in blue, noisy traces in orange

- [x] Add legend per subplot

- [x] Add signal formula as subplot title

- [x] Add grid to all subplots

- [x] Set figure title "Clean vs Noisy Signals"

- [x] Tight layout applied

- [x] Save as `results/noisy_vs_clean.png` at 300 DPI

- [x] Verify file saved correctly



---



## Phase 22 — Visualisation: `results/noise_distribution.png`



- [x] Compute N_amp for all 4 signals

- [x] Compute N_phase for all 4 signals

- [x] Create figure with 2 rows × 4 columns (row=type, col=signal)

- [x] Row 1: N_amp histograms per signal (s1–s4)

- [x] Row 2: N_phase histograms per signal (s1–s4)

- [x] Overlay fitted Gaussian PDF curve on each histogram

- [x] Display measured σ and configured σ in each plot

- [x] Add x-axis label "Noise Value"

- [x] Add y-axis label "Density"

- [x] Add subplot titles "N_amp s1", "N_amp s2", etc.

- [x] Highlight burst-noise contribution separately (different colour bar)

- [x] Set figure title "Noise Distribution Analysis"

- [x] Tight layout applied

- [x] Save as `results/noise_distribution.png` at 300 DPI

- [x] Verify file saved correctly



---



## Phase 23 — Visualisation: `results/frequency_spectrum.png`



- [x] Compute FFT of each clean signal (s1–s5)

- [x] Compute FFT of each noisy signal (s1–s5)

- [x] Create figure with 5 rows × 2 columns (clean FFT | noisy FFT)

- [x] Plot magnitude spectrum (positive frequencies only) for each

- [x] Add vertical dashed line at expected dominant frequency for s1–s4

- [x] Add x-axis label "Frequency (Hz)"

- [x] Add y-axis label "Magnitude"

- [x] Set x-axis limit to [0, 200] Hz

- [x] Use log scale on y-axis for better visibility

- [x] Add subplot titles with signal ID and frequency

- [x] Add grid to all subplots

- [x] Set figure title "Frequency Spectrum: Clean vs Noisy"

- [x] Tight layout applied

- [x] Save as `results/frequency_spectrum.png` at 300 DPI

- [x] Verify file saved correctly



---



## Phase 24 — Visualisation: `results/burst_events.png`



- [x] Select signal s1 for burst demonstration

- [x] Plot clean s1 (first 1.0 second)

- [x] Plot noisy s1 on same axis

- [x] Shade burst-event regions in red with alpha=0.3

- [x] Add horizontal dashed lines at ±(A + burst_magnitude) to show expected extremes

- [x] Add legend: "Clean", "Noisy", "Burst region"

- [x] Add x-axis label "Time (s)"

- [x] Add y-axis label "Amplitude"

- [x] Add title "Burst Noise Events on s1"

- [x] Add grid

- [x] Create second subplot: N_amp over time with burst events highlighted

- [x] Save as `results/burst_events.png` at 300 DPI

- [x] Verify file saved correctly



---



## Phase 25 — Visualisation: `results/dataset_splits.png`



- [x] Create bar chart showing train/val/test sample counts

- [x] Add percentage labels above each bar

- [x] Add x-axis labels: Train, Validation, Test

- [x] Add y-axis label "Number of Windows"

- [x] Add title "Dataset Split Distribution"

- [x] Add grid on y-axis

- [x] Add text box with total window count

- [x] Save as `results/dataset_splits.png` at 300 DPI

- [x] Verify file saved correctly



---



## Phase 26 — Visualisation: `results/context_window_example.png`



- [x] Select 5 example windows from the dataset (one for each possible C index 0–4)

- [x] For each window create 2 subplots side-by-side:

  - [x] Left: line plot of x_w (composite noisy window, 10 samples)

  - [x] Right: bar chart of C vector (5 bars, signal labels s1–s5 on x-axis)

- [x] Highlight the selected bar (C[i]=1) in a distinct colour (e.g. red), rest grey

- [x] Add text annotation on left plot: "Label: y = {value:.3f} ({signal_id}_clean)"

- [x] Add x-axis label "Sample index within window" on left subplot

- [x] Add x-axis label "Signal" on right subplot (s1, s2, s3, s4, s5)

- [x] Add y-axis label "Amplitude" on left subplot

- [x] Add y-axis label "C value (0 or 1)" on right subplot

- [x] Add subplot row titles "Window k=... → extract s{i+1}"

- [x] Set figure title "Context Window + One-Hot Signal Selector Examples"

- [x] Tight layout applied

- [x] Save as `results/context_window_example.png` at 300 DPI

- [x] Verify file saved correctly

- [x] Verify NO "x_feature" label appears anywhere in the plot (x_feature is obsolete)



---



## Phase 27 — Jupyter Notebook: `notebooks/dataset_analysis.ipynb`



- [x] Create `notebooks/dataset_analysis.ipynb`

- [x] Cell 1 — Title markdown: "# Signal Dataset Analysis"

- [x] Cell 2 — Imports: numpy, matplotlib, seaborn, scipy, pathlib, sys

- [x] Cell 3 — Set sys.path to include src/

- [x] Cell 4 — Import DatasetSDK, instantiate, call generate_dataset()

- [x] Cell 5 — Markdown: "## 1. Signal Overview"

- [x] Cell 6 — Load clean signals from raw .npz

- [x] Cell 7 — Plot all 5 clean signals (time domain, first 0.5 sec)

- [x] Cell 8 — Print signal statistics: min, max, mean, std per signal

- [x] Cell 9 — Markdown: "## 2. Noise Characterisation"

- [x] Cell 10 — Compute N_amp per signal

- [x] Cell 11 — Plot N_amp histograms with Gaussian fit overlaid

- [x] Cell 12 — Compute and print: measured σ vs configured σ for each signal

- [x] Cell 13 — Compute N_phase per signal

- [x] Cell 14 — Plot N_phase histograms with Gaussian fit overlaid

- [x] Cell 15 — Identify burst events: print count, mean duration, max duration

- [x] Cell 16 — Markdown: "## 3. Frequency Analysis"

- [x] Cell 17 — Compute FFT of clean signals

- [x] Cell 18 — Plot FFT magnitude spectrum for all 5 clean signals

- [x] Cell 19 — Verify dominant frequency peaks match specification (table)

- [x] Cell 20 — Compute FFT of noisy signals

- [x] Cell 21 — Plot FFT comparison: clean vs noisy for each base signal

- [x] Cell 22 — Markdown: "## 4. Dataset Split Statistics"

- [x] Cell 23 — Load dataset.npz, print shapes of all **9** arrays (X_train, C_train, y_train, X_val, C_val, y_val, X_test, C_test, y_test)

- [x] Cell 24 — Print train/val/test sizes and percentages; verify they match config ratios (70/15/15)

- [x] Cell 25 — Check for NaN/Inf in all 9 arrays — print PASS/FAIL per array

- [x] Cell 26 — Plot distribution of y_train values; group by selected signal (argmax(C_train))

- [x] Cell 27 — Markdown: "## 5. Context Window Visualisation"

- [x] Cell 28 — Sample 10 windows from X_train; for each show x_w (line) and the C-selected signal label

- [x] Cell 29 — Visualise C_train vectors as heatmap (**10 windows × 5 signal classes**); x-axis: s1–s5

- [x] Cell 30 — Compute frequency of each signal index in C_train; verify approximately uniform over {0,1,2,3,4}

- [x] Cell 31 — Markdown: "## 6. Summary & Conclusions"

- [x] Cell 32 — Table: signal specs vs measured properties

- [x] Cell 33 — Table: noise parameters configured vs measured

- [x] Cell 34 — Table: dataset shape summary

- [x] Cell 35 — Conclusions text on readiness for model training phase

- [x] Run all cells — confirm no errors

- [x] Verify all plots render inline

- [x] Save notebook with output included (`jupyter nbconvert --to notebook --execute`)



---



## Phase 28 — README.md



- [x] Create `README.md` at project root

- [x] Add project title: "Noisy Sine Signal Dataset Generator"

- [x] Add one-paragraph overview of the project purpose

- [x] Add "## Signals" section with table of all 5 signals and formulas

- [x] Add "## Noise Model" section explaining Gaussian + burst noise

- [x] Add "## Dataset Structure" section explaining context window and labels

- [x] Add "## Quickstart" section

- [x] Add step: clone/download repo

- [x] Add step: `uv sync`

- [x] Add step: `uv run python -m signal_dataset`

- [x] Add step: explain output files (`data/dataset.npz`, `data/signals_raw.npz`)

- [x] Add "## Project Structure" section with annotated directory tree

- [x] Add "## Configuration" section explaining `config/setup.json` fields

- [x] Add "## Testing" section: `uv run pytest tests/`, coverage badge info

- [x] Add "## Linting" section: `uv run ruff check`

- [x] Add "## Results" section with embedded visualisation images

- [x] Embed `results/clean_signals.png` in README

- [x] Embed `results/noisy_vs_clean.png` in README

- [x] Embed `results/frequency_spectrum.png` in README

- [x] Add "## Next Phase" section mentioning RNN / FC / LSTM training

- [x] Add "## License" section

- [x] Add "## Author" section with contact email

- [x] Verify README renders correctly as markdown

- [x] Verify all image paths are correct relative to repo root



---



## Phase 29 — `docs/PRD_dataset_generation.md`



- [x] Create `docs/PRD_dataset_generation.md`

- [x] Add title and version header

- [x] Add "Algorithm: Dataset Generation Pipeline" section

- [x] Document INPUT DATA: config paths, signal parameters, noise parameters

- [x] Document OUTPUT DATA: dataset.npz keys and shapes, signals_raw.npz keys and shapes

- [x] Document SETUP DATA: all configurable parameters with defaults and valid ranges

- [x] Write out the full noise injection algorithm step-by-step

- [x] Write out the full windowing algorithm step-by-step

- [x] Write out the full chronological split algorithm

- [x] Add pseudocode for the complete pipeline

- [x] Add parameter sensitivity analysis plan (OAT): vary sigma_amp, sigma_phase, window_size, burst_probability

- [x] List expected effects of each parameter change

- [x] Document known edge cases and how they are handled

- [x] Add references / citations for Gaussian white noise and burst noise models

- [x] Submit for approval



---



## Phase 30 — Prompt Engineering Log



- [x] Create `docs/prompt_engineering_log.md`

- [x] Add header: "# Prompt Engineering Log"

- [x] Document Prompt 1: initial project description provided to AI

- [x] Document Prompt 2: PRD/PLAN/TODO generation

- [x] Document AI model used: claude-sonnet-4-6

- [x] Document date of each prompt

- [x] Document outcome of each prompt (what was generated)

- [x] Document any iterative refinements made

- [x] Document any corrections applied to AI output

- [x] Keep log updated as development proceeds



---



## Phase 31 — Final Submission Checklist



- [x] All docs present: PRD.md, PLAN.md, TODO.md, PRD_dataset_generation.md, README.md

- [x] Prompt Engineering Log complete

- [x] `uv run ruff check` → 0 errors across entire project

- [x] `uv run pytest tests/ --cov=src` → ≥ 85% coverage

- [x] All source files ≤ 150 lines

- [x] All test files ≤ 150 lines

- [x] No hardcoded signal parameters in Python source

- [x] No API keys or secrets in source

- [x] `.env-example` committed with dummy values

- [x] `.env` in `.gitignore` and not tracked

- [x] `uv.lock` committed and up to date

- [x] `pyproject.toml` is sole dependency declaration

- [x] No `requirements.txt` present

- [x] No `pip install` used anywhere

- [x] `data/dataset.npz` generated and loadable

- [x] `data/signals_raw.npz` generated and loadable — contains all 5 clean + 5 noisy vectors

- [x] All **7** visualisation `.png` files present in `results/`: clean_signals, noisy_vs_clean, noise_distribution, frequency_spectrum, burst_events, dataset_splits, context_window_example

- [x] dataset.npz contains all 9 arrays: X_train, C_train, y_train, X_val, C_val, y_val, X_test, C_test, y_test

- [x] C arrays in dataset.npz are valid one-hot (every row sums to 1, second dim = 5)

- [x] y arrays in dataset.npz have shape (N_split, 1) — scalar per sample, not multi-output

- [x] X arrays sourced from s5_noisy only — verified by comparing to raw npz

- [x] Running generator twice with seed 42 produces byte-identical dataset.npz (reproducibility)

- [x] `notebooks/dataset_analysis.ipynb` executed end-to-end without errors (`jupyter nbconvert --execute`)

- [x] All notebook cells have output saved

- [x] `DatasetSDK` is sole entry point confirmed (no service imported directly in main.py)

- [x] `ApiGatekeeper` wired through all future external calls

- [x] `ConfigManager` used for all config reads

- [x] `version.py` reflects correct version string "1.00"

- [x] Git history has structured commit messages

- [x] Each logical unit committed separately (scaffold, config, services, tests, docs)

- [x] README.md renders with correct images

- [x] Integration test passes on clean environment (`uv sync` → `uv run pytest`)

- [x] Model training phase (RNN/FC/LSTM) confirmed as out of scope for this submission

- [x] Supervisor final review completed

- [x] Submission packaged and delivered



---



## Blocked / Notes



- Model training (RNN / FC / LSTM) is out of scope for this phase

- Supervisor approval required before Phase 1 coding begins

- `data/` and `results/` are git-ignored; regenerate locally with `uv run python -m signal_dataset`

- Burst noise implementation must use `np.random.default_rng` (not legacy `np.random`) for reproducibility




