# PLAN — Architecture & Design

**Version:** 1.00
**Date:** 2026-05-10
**Status:** Awaiting Approval

---

## 1. Technology Stack

| Concern | Choice | Reason |
|---------|--------|--------|
| Language | Python 3.10+ | Required by guidelines |
| Package manager | uv | Mandatory per CLAUDE.md |
| Linter | Ruff | Mandatory per CLAUDE.md |
| Numerics | NumPy | Vectorised signal ops |
| Visualisation | Matplotlib + Seaborn | Standard, offline |
| Notebooks | Jupyter | Required by guidelines |
| Testing | pytest + pytest-cov | TDD mandatory |

---

## 2. Repository Structure

```
HW1/
├── src/signal_dataset/
│   ├── __init__.py              # exports __all__, __version__
│   ├── sdk/
│   │   └── sdk.py               # Single entry point for all logic
│   ├── services/
│   │   ├── signal_generator.py  # Clean sine wave generation
│   │   ├── noise_injector.py    # Gaussian + burst noise injection
│   │   ├── windower.py          # Sliding window + one-hot C vector
│   │   └── dataset_builder.py   # Assembly, splits, .npz save
│   ├── shared/
│   │   ├── gatekeeper.py        # ApiGatekeeper (required by spec)
│   │   ├── config.py            # ConfigManager — loads setup.json
│   │   └── version.py           # __version__ = "1.00"
│   ├── constants.py             # Physical/math constants only
│   └── main.py                  # CLI entry point
├── tests/
│   ├── unit/
│   │   ├── test_signal_generator.py
│   │   ├── test_noise_injector.py
│   │   ├── test_windower.py
│   │   ├── test_dataset_builder.py
│   │   ├── test_config.py
│   │   └── test_sdk.py
│   ├── integration/
│   │   └── test_pipeline.py     # end-to-end dataset generation
│   └── conftest.py
├── config/
│   ├── setup.json               # All dataset parameters
│   └── rate_limits.json         # Gatekeeper rate limits
├── data/                        # Generated — git-ignored
├── results/                     # Charts — git-ignored
├── assets/
├── notebooks/
│   └── dataset_analysis.ipynb
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── TODO.md
│   └── PRD_dataset_generation.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env-example
└── .gitignore
```

---

## 3. Module Responsibilities

### 3.1 `sdk/sdk.py` — `DatasetSDK`
Single public interface. All external code calls only this.

```
DatasetSDK
  └── generate_dataset(config_path) -> DatasetResult
        ├── ConfigManager.load()
        ├── SignalGenerator.generate_all()
        ├── NoiseInjector.inject_all()
        ├── Windower.build_windows()
        └── DatasetBuilder.assemble_and_save()
```

### 3.2 `services/signal_generator.py` — `SignalGenerator`
- **Input**: signal configs (list of amp/freq/phase), time axis
- **Output**: `Dict[str, np.ndarray]` — 5 clean signal vectors `(10 000,)`
- **Logic**: vectorised NumPy sine evaluation + composite sum

### 3.3 `services/noise_injector.py` — `NoiseInjector`
- **Input**: clean signal vectors, noise config
- **Output**: `Dict[str, np.ndarray]` — 5 noisy signal vectors `(10 000,)`
- **Logic**:
  - `N_amp = gaussian_amp + burst_amp`
  - `N_phase = gaussian_phase + burst_phase`
  - `s_noisy = (A + N_amp) · sin(2π·f·t + φ + N_phase)`
  - Burst events generated with a Poisson-like random mask

### 3.4 `services/windower.py` — `Windower`
- **Input**: noisy signal vectors (all 5), clean signal vectors (all 5), window size W, random seed
- **Output**:
  - `X  (N_windows, W)` — composite noisy windows (source: `s5_noisy` only)
  - `C  (N_windows, 5)` — one-hot signal selectors (which of s1–s5 to extract)
  - `y  (N_windows, 1)` — scalar clean label for the C-selected signal at window center
- **Logic**:
  - Stride-1 sliding window over `s5_noisy` → `x_w = s5_noisy[k : k+W]`
  - Per window: sample `i ~ Uniform({0,1,2,3,4})`, build `C = one_hot(i)` — shape `(5,)`
  - Label: `y = clean_signals[i][k + W//2]` — scalar, the clean value of the chosen signal
  - Individual noisy signals (s1–s4) are **not** part of the model input; only s5_noisy is used

### 3.5 `services/dataset_builder.py` — `DatasetBuilder`
- **Input**: `X (N, W)`, `C (N, 5)`, `y (N, 1)` from Windower; split ratios
- **Output**: `data/dataset.npz`, `data/signals_raw.npz`
- **Logic**: chronological split (no shuffle), save `X`, `C`, `y` per split as compressed npz
- `.npz` keys per split: `X_train`, `C_train`, `y_train`, `X_val`, `C_val`, `y_val`, `X_test`, `C_test`, `y_test`

### 3.6 `shared/config.py` — `ConfigManager`
- Loads and validates `config/setup.json` and `config/rate_limits.json`
- Exposes typed dataclasses for each config section
- Raises `ConfigValidationError` on bad values

### 3.7 `shared/gatekeeper.py` — `ApiGatekeeper`
- FIFO queue for all external calls
- Rate-limit enforcement from `rate_limits.json`
- Required by spec; used here for any future external data or model API calls

---

## 4. Data Flow

```
config/setup.json
      │
      ▼
  ConfigManager
      │
      ▼
SignalGenerator ──► s1_clean … s5_clean  shape (10 000,) each
      │
      ▼
NoiseInjector ───► s1_noisy … s5_noisy  shape (10 000,) each
      │                │
      │           s5_noisy = s1_noisy + s2_noisy + s3_noisy + s4_noisy
      │
      ▼
Windower ─────────► X  (N_windows, W)   — windows from s5_noisy ONLY
      │              C  (N_windows, 5)   — one-hot signal selector
      │              y  (N_windows, 1)   — scalar: clean value of C-selected signal
      ▼
DatasetBuilder ───► data/dataset.npz    (X_train, C_train, y_train, …)
                    data/signals_raw.npz
                    results/*.png
```

---

## 5. Noise Model Detail

```
For signal i with amplitude A_i, frequency f_i, phase φ_i:

  g_amp(t)   ~ N(0, σ_amp_i²)          # per-sample Gaussian
  b_amp(t)   = M(t) · U(amp_mag)        # burst mask × uniform magnitude
  N_amp(t)   = g_amp(t) + b_amp(t)

  g_phase(t) ~ N(0, σ_phase_i²)
  b_phase(t) = M(t) · U(phase_mag)
  N_phase(t) = g_phase(t) + b_phase(t)

  s_noisy_i(t) = (A_i + N_amp(t)) · sin(2π · f_i · t + φ_i + N_phase(t))

Burst mask M(t): binary array; bursts placed with probability p per sample,
each burst lasting duration ~ Uniform(d_min, d_max) samples.
```

---

## 6. Context Window Detail

```
For window starting at index k:

  ── INPUT ────────────────────────────────────────────────────────────
  x_w = s5_noisy[k : k+W]          shape (W,)   ← composite noisy ONLY

  i   ~ Uniform({0, 1, 2, 3, 4})                ← random signal index
  C   = one_hot(i)                  shape (5,)   ← signal selector
        C[j] = 1 if j == i, else 0

  At model training time the full input is formed by concatenation:
  x_input = concat([x_w, C])        shape (W+5,) = (15,)

  ── LABEL ────────────────────────────────────────────────────────────
  signals_clean = [s1_clean, s2_clean, s3_clean, s4_clean, s5_clean]

  y = signals_clean[i][k + W//2]   shape (1,)   ← scalar clean value
                                                    of the selected signal

  ── STORAGE IN .npz ──────────────────────────────────────────────────
  X[k]  = x_w          (W,)
  C[k]  = C            (5,)          stored separately — NOT concatenated
  y[k]  = y            (1,)

  Model concatenates X and C itself at forward-pass time.

  ── EXAMPLE ──────────────────────────────────────────────────────────
  k=0, i=1 (s2 selected):
    x_w  = s5_noisy[0:10]           10 composite noisy samples
    C    = [0, 1, 0, 0, 0]          "extract s2"
    y    = s2_clean[5]              clean s2 value at sample 5
```

---

## 7. File Size Budget (≤ 150 lines each)

| File | Est. lines |
|------|-----------|
| sdk.py | ~80 |
| signal_generator.py | ~90 |
| noise_injector.py | ~120 |
| windower.py | ~90 |
| dataset_builder.py | ~100 |
| config.py | ~100 |
| gatekeeper.py | ~120 |
| main.py | ~60 |
| constants.py | ~30 |

All under 150 — no splits needed at this scope.

---

## 8. Testing Strategy (TDD)

| Test file | What it covers |
|-----------|---------------|
| `test_signal_generator.py` | Correct amplitudes, frequencies, phases; composite = sum |
| `test_noise_injector.py` | Noise shape, Gaussian distribution params, burst mask properties |
| `test_windower.py` | Window count, one-hot validity, feature shape |
| `test_dataset_builder.py` | Split sizes, npz keys, shape correctness |
| `test_config.py` | Load, validate, reject bad config |
| `test_sdk.py` | End-to-end DatasetSDK call returns correct result |
| `test_pipeline.py` (integration) | Full pipeline: config → .npz files exist |

Target: ≥ 85% coverage enforced by `fail_under = 85`.

---

## 9. Visualisations (saved to `results/`)

1. `results/clean_signals.png` — all 5 clean signals, time-domain overlay
2. `results/noisy_vs_clean.png` — side-by-side per signal (clean vs noisy)
3. `results/noise_distribution.png` — histogram of N_amp and N_phase per signal
4. `results/frequency_spectrum.png` — FFT magnitude spectrum (clean + noisy)
5. `results/burst_events.png` — burst mask overlay on one signal
