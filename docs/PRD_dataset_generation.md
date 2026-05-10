# PRD — Dataset Generation Algorithm

**Version:** 1.00
**Date:** 2026-05-10
**Status:** Complete

---

## Algorithm: Dataset Generation Pipeline

### Input Data
| Item | Type | Description |
|------|------|-------------|
| `config/setup.json` | JSON file | All dataset parameters |
| `config/rate_limits.json` | JSON file | API gatekeeper rate limits |

Signal parameters (per signal): `id`, `amplitude`, `frequency_hz`, `phase_rad`
Noise parameters: `sigma_amp[]`, `sigma_phase[]`, `burst.probability`, `duration_range_samples`, `amp_magnitude`, `phase_magnitude`
Dataset parameters: `duration_sec`, `sample_rate_hz`, `window_size`, `train_ratio`, `val_ratio`, `test_ratio`, `random_seed`

---

### Output Data

**`data/dataset.npz`**

| Key | Shape | Description |
|-----|-------|-------------|
| `X_train` | `(N_train, W)` | Composite noisy windows — training set |
| `C_train` | `(N_train, 5)` | One-hot signal selectors — training set |
| `y_train` | `(N_train, 1)` | Scalar clean labels — training set |
| `X_val` | `(N_val, W)` | Validation windows |
| `C_val` | `(N_val, 5)` | Validation selectors |
| `y_val` | `(N_val, 1)` | Validation labels |
| `X_test` | `(N_test, W)` | Test windows |
| `C_test` | `(N_test, 5)` | Test selectors |
| `y_test` | `(N_test, 1)` | Test labels |

**`data/signals_raw.npz`**

| Key | Shape | Description |
|-----|-------|-------------|
| `clean_signals` | `(5, 10000)` | Rows: s1_clean, s2_clean, s3_clean, s4_clean, s5_clean |
| `noisy_signals` | `(5, 10000)` | Rows: s1_noisy, s2_noisy, s3_noisy, s4_noisy, s5_noisy |
| `time_axis` | `(10000,)` | Time axis: t ∈ [0, 9.999] seconds |

---

### Setup Data

| Parameter | Default | Valid Range | Description |
|-----------|---------|-------------|-------------|
| `duration_sec` | 10 | > 0 | Total signal duration |
| `sample_rate_hz` | 1000 | > 0 | Sampling frequency |
| `window_size` | 10 | > 0 | Context window size W |
| `train_ratio` | 0.70 | (0, 1), sum=1 | Train fraction |
| `val_ratio` | 0.15 | (0, 1), sum=1 | Validation fraction |
| `test_ratio` | 0.15 | (0, 1), sum=1 | Test fraction |
| `random_seed` | 42 | integer | RNG seed for reproducibility |
| `sigma_amp[i]` | [0.05,0.05,0.03,0.02] | ≥ 0 | Gaussian amp noise σ per signal |
| `sigma_phase[i]` | [0.05,0.05,0.03,0.02] | ≥ 0 | Gaussian phase noise σ per signal |
| `burst.probability` | 0.01 | [0, 1] | Per-sample burst onset probability |
| `burst.duration_range` | [5, 20] | lo < hi > 0 | Burst duration in samples |
| `burst.amp_magnitude` | 0.5 | ≥ 0 | Burst amplitude magnitude |
| `burst.phase_magnitude` | 0.3 | ≥ 0 | Burst phase magnitude |

---

## Algorithm Steps

### Step 1 — Time Axis Generation
```
t = linspace(0, duration_sec, n_samples, endpoint=False)
n_samples = duration_sec × sample_rate_hz = 10000
```

### Step 2 — Clean Signal Generation
```
For each signal i in {s1, s2, s3, s4}:
    si_clean(t) = A_i × sin(2π × f_i × t + φ_i)
s5_clean = s1_clean + s2_clean + s3_clean + s4_clean
```

### Step 3 — Noise Injection
```
For each signal i:
    g_amp(t)   ~ N(0, σ_amp_i²)         per sample
    b_amp(t)   = burst_mask(t) × U(-amp_mag, +amp_mag)
    N_amp(t)   = g_amp(t) + b_amp(t)

    g_phase(t) ~ N(0, σ_phase_i²)
    b_phase(t) = burst_mask(t) × U(-phase_mag, +phase_mag)
    N_phase(t) = g_phase(t) + b_phase(t)

    si_noisy(t) = (A_i + N_amp(t)) × sin(2π × f_i × t + φ_i + N_phase(t))

s5_noisy = s1_noisy + s2_noisy + s3_noisy + s4_noisy
```

**Burst Mask Generation:**
```
i = 0
while i < n_samples:
    if random() < p_burst:
        dur = randint(d_min, d_max)   # inclusive bounds
        mask[i : i + dur] = 1
        i += dur
    else:
        i += 1
```

### Step 4 — Sliding Window Construction
```
N_windows = n_samples - W + 1 = 9991

For k in 0..N_windows-1:
    X[k] = s5_noisy[k : k+W]          shape (W,)   -- composite noisy only
    i     ~ Uniform({0,1,2,3,4})
    C[k]  = one_hot(i)                 shape (5,)   -- signal selector
    y[k]  = clean_signals[i][k + W//2] shape (1,)   -- scalar label
```

### Step 5 — Chronological Split
```
n_train = int(N_windows × train_ratio) = 6993
n_val   = int(N_windows × val_ratio)   = 1498
n_test  = N_windows - n_train - n_val  = 1500

X_train = X[0 : n_train]
X_val   = X[n_train : n_train + n_val]
X_test  = X[n_train + n_val :]
(same indices applied to C and y)
```

---

## Pseudocode — Complete Pipeline

```python
cfg = ConfigManager.load()
t = linspace(0, cfg.duration_sec, n_samples, endpoint=False)

clean = {sid: A*sin(TWO_PI*f*t + phi) for each signal}
clean["s5"] = sum of s1..s4

noisy = {sid: (A + N_amp) * sin(TWO_PI*f*t + phi + N_phase) for each signal}
noisy["s5"] = sum of noisy s1..s4

for k in 0..N-1:
    X[k] = noisy["s5"][k : k+W]
    i = random choice from {0,1,2,3,4}
    C[k] = one_hot(i)
    y[k] = clean[SIGNAL_IDS[i]][k + W//2]

splits = chronological_split(X, C, y, ratios)
save dataset.npz with X_train/C_train/y_train + val + test
save signals_raw.npz with clean_signals, noisy_signals, time_axis
```

---

## Parameter Sensitivity Analysis (OAT)

| Parameter | Varied Range | Expected Effect |
|-----------|-------------|-----------------|
| `sigma_amp` | 0 → 0.5 | Higher σ → noisier amplitude envelope, harder separation |
| `sigma_phase` | 0 → 0.5 | Higher σ → frequency smearing in noisy signals |
| `window_size` W | 5 → 50 | Larger W → more temporal context, more windows |
| `burst.probability` | 0 → 0.05 | Higher p → more burst events, heavier tails |
| `burst.amp_magnitude` | 0.1 → 2.0 | Larger → more extreme outliers |

---

## Known Edge Cases

| Case | Handling |
|------|---------|
| Signal shorter than W | `ValueError` raised in `Windower.build_windows()` |
| Burst run extends past signal end | Clamped to `n_samples` boundary |
| Sigma = 0 | Returns zero-array (no Gaussian noise component) |
| Burst probability = 0 | Returns all-zeros mask (no burst events) |
| Split ratios don't sum to 1 | `ConfigValidationError` raised in `ConfigManager._validate()` |

---

## References

- Proakis, J. & Manolakis, D. (2007). *Digital Signal Processing*. Pearson.
- Gaussian white noise model: standard AWGN as in communications literature.
- Burst noise model: derived from impulse noise models in power systems (Middleton Class A).
