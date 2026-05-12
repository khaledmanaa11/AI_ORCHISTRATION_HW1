# PRD — Noisy Sine Signal Dataset Generator

**Version:** 1.00
**Date:** 2026-05-10
**Author:** Khaled Mnaa
**Status:** Awaiting Approval

---

## 1. Purpose

Build a labeled dataset from four pure sine waves and their composite sum, each corrupted by
amplitude and phase noise (Gaussian white noise + burst noise), windowed with a one-hot context
selector. The dataset will serve as training/evaluation data for RNN, FC, and LSTM models whose
task is to recover the original clean signal from the noisy composite.

---

## 2. Background & Motivation

Signal separation under noise is a core problem in time-series learning. Using analytically defined
sine components with controlled noise allows precise ground-truth comparison and reproducible
benchmarking across neural architectures.

The four frequencies (5 Hz, 15 Hz, 50 Hz, 100 Hz) were selected because they span two decades of
frequency range, have distinct amplitude profiles, and will stress-test the frequency selectivity of
RNN/LSTM models.

---

## 3. Signals

| ID | Formula | Amplitude | Frequency (Hz) | Phase (rad) |
|----|---------|-----------|----------------|-------------|
| s1 | 2.0 · sin(2π · 5 · t)         | 2.0 | 5   | 0     |
| s2 | 1.5 · sin(2π · 15 · t + π/4)  | 1.5 | 15  | π/4   |
| s3 | 0.8 · sin(2π · 50 · t)         | 0.8 | 50  | 0     |
| s4 | 0.3 · sin(2π · 100 · t)        | 0.3 | 100 | 0     |
| s5 | s1 + s2 + s3 + s4 (composite)  | —   | —   | —     |

---

## 4. Dataset Generation Pipeline

### 4.1 Time Axis
- Duration: 10 seconds
- Sampling frequency: 1 000 Hz
- Total samples: 10 000 per signal
- `t = [0, 0.001, 0.002, …, 9.999]` (shape: `(10 000,)`)

### 4.2 Clean Signal Vectors
Five vectors of shape `(10 000,)`:
- `s1_clean, s2_clean, s3_clean, s4_clean` — individual components
- `s5_clean = s1 + s2 + s3 + s4` — composite

### 4.3 Noise Model
For each signal `si`, the noisy version is:

```
s_noisy(t) = (A_i + N_amp(t)) · sin(2π · f_i · t + φ_i + N_phase(t))
```

Where the noise terms are a combination of:

| Term | Type | Description |
|------|------|-------------|
| `N_amp` | Gaussian white noise | `σ_amp` configurable per signal |
| `N_amp` | Burst noise | Random-duration additive amplitude spikes |
| `N_phase` | Gaussian white noise | `σ_phase` configurable per signal |
| `N_phase` | Burst noise | Random-duration phase perturbations |

Noise parameters come from `config/setup.json` — never hardcoded.

### 4.4 Context Window + One-Hot Signal Selector

**Step 1 — Window from composite noisy signal only:**
- Window size: `W = 10` samples (configurable)
- The input window is always extracted from **`s5_noisy`** (the composite of all four noisy components):
  `x_w = s5_noisy[t_i : t_i+W]` — shape `(W,)`
- The individual noisy signals are **never** directly exposed to the model as input

**Step 2 — Signal selector vector `C`:**
- `C` is a random one-hot vector of length **5** (one entry per signal: s1, s2, s3, s4, s5):
  `C = one_hot(i)` where `i ~ Uniform({0, 1, 2, 3, 4})` — shape `(5,)`
- `C[i] = 1` means "extract the clean version of signal `i` from this window"
- `C` acts as a conditioning signal telling the model which component to isolate

**Step 3 — Dataset sample construction:**
- Each dataset sample is the pair `(x_w, C)` stored separately:
  - `X_window[k] = x_w` — shape `(W,)` — the composite noisy window
  - `C[k]` — shape `(5,)` — the one-hot signal selector
- At model training time, `x_w` and `C` are concatenated to form the full input:
  `x_input = concat([x_w, C])` — shape `(W + 5,)` = `(15,)`

### 4.5 Labels (Ground Truth)
- The label is the **scalar** clean value of the **C-selected signal** at the window center:
  `y[k] = clean_signals[argmax(C[k])][t_i + W//2]` — shape `(1,)` (scalar per sample)
- Because `C` is known at dataset-build time, we can always retrieve the exact ground truth
- Example: if `C = [0, 1, 0, 0, 0]` → label = `s2_clean[t_i + W//2]`

### 4.6 Dataset Output
| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | 70% | Model training |
| Val   | 15% | Hyperparameter tuning |
| Test  | 15% | Final evaluation |

Saved as:
- `data/dataset.npz` — compressed numpy arrays:
  - `X_train`, `X_val`, `X_test` — composite noisy windows, shape `(N_split, W)`
  - `C_train`, `C_val`, `C_test` — one-hot signal selectors, shape `(N_split, 5)`
  - `y_train`, `y_val`, `y_test` — scalar labels (clean value of selected signal), shape `(N_split, 1)`
- `data/signals_raw.npz` — all clean and noisy signal vectors for analysis

---

## 5. Configuration

All parameters in `config/setup.json`:

```json
{
  "dataset": {
    "duration_sec": 10,
    "sample_rate_hz": 1000,
    "window_size": 10,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42
  },
  "signals": [
    {"id": "s1", "amplitude": 2.0, "frequency_hz": 5,   "phase_rad": 0},
    {"id": "s2", "amplitude": 1.5, "frequency_hz": 15,  "phase_rad": 0.7854},
    {"id": "s3", "amplitude": 0.8, "frequency_hz": 50,  "phase_rad": 0},
    {"id": "s4", "amplitude": 0.3, "frequency_hz": 100, "phase_rad": 0}
  ],
  "noise": {
    "gaussian": {
      "sigma_amp":   [0.05, 0.05, 0.03, 0.02],
      "sigma_phase": [0.05, 0.05, 0.03, 0.02]
    },
    "burst": {
      "probability": 0.01,
      "duration_range_samples": [5, 20],
      "amp_magnitude": 0.5,
      "phase_magnitude": 0.3
    }
  }
}
```

---

## 6. Deliverables
1. `data/dataset.npz` — train/val/test splits ready for model ingestion
2. `data/signals_raw.npz` — raw clean and noisy vectors
3. `results/` — visualization plots (time-domain, frequency-domain, noise distribution)
4. `notebooks/dataset_analysis.ipynb` — exploratory analysis notebook
5. Full test suite with ≥ 85% coverage

---

## 7. Out of Scope (this phase)
- RNN / FC / LSTM model training (next phase)
- Real-time signal streaming
- Non-sine waveforms

---

## 8. Acceptance Criteria
- [ ] All 5 clean signals generated with correct amplitudes, frequencies, and phases
- [ ] Noisy signals correctly combine Gaussian + burst noise on both amplitude and phase
- [ ] Windows extracted exclusively from `s5_noisy` (composite noisy signal)
- [ ] `C` vectors are one-hot of length **5**, selecting a signal index (not a window position)
- [ ] Labels are scalars: `clean_signals[argmax(C)][center_index]`
- [ ] Dataset `.npz` stores `X`, `C`, `y` arrays separately per split
- [ ] `uv run ruff check` → 0 errors
- [ ] `uv run pytest tests/` → ≥ 85% coverage
- [ ] At least 3 visualizations saved to `results/`
