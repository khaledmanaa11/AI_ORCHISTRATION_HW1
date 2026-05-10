# Noisy Sine Signal Dataset Generator

A Python package that generates a labeled dataset from four pure sine waves and their composite,
each corrupted by amplitude and phase noise (Gaussian white noise + burst noise), windowed with
a one-hot context selector. The dataset is ready for training RNN, FC, and LSTM models.

---

## Signals

| ID | Formula | Amplitude | Frequency (Hz) | Phase (rad) |
|----|---------|-----------|----------------|-------------|
| s1 | 2.0 · sin(2π · 5 · t) | 2.0 | 5 | 0 |
| s2 | 1.5 · sin(2π · 15 · t + π/4) | 1.5 | 15 | π/4 |
| s3 | 0.8 · sin(2π · 50 · t) | 0.8 | 50 | 0 |
| s4 | 0.3 · sin(2π · 100 · t) | 0.3 | 100 | 0 |
| s5 | s1 + s2 + s3 + s4 (composite) | — | — | — |

---

## Noise Model

For each signal `si`, the noisy version is:

```
s_noisy(t) = (A_i + N_amp(t)) · sin(2π · f_i · t + φ_i + N_phase(t))
```

Where:
- `N_amp = gaussian_amp + burst_amp` — amplitude perturbation
- `N_phase = gaussian_phase + burst_phase` — phase perturbation
- Gaussian: per-sample Gaussian white noise with configurable σ per signal
- Burst: random-duration amplitude/phase spikes (probability, duration, magnitude configurable)

---

## Dataset Structure

- **X** `(N, W)` — composite noisy windows extracted from `s5_noisy` only
- **C** `(N, 5)` — one-hot signal selector (which of s1–s5 to isolate)
- **y** `(N, 1)` — scalar clean value of the C-selected signal at the window center

At model training time: `x_input = concat([x_w, C])` → shape `(W+5,)` = `(15,)`

---

## Quickstart

```bash
# 1. Clone or download repo
git clone <repo-url>
cd HW1

# 2. Install dependencies
uv sync

# 3. Generate dataset
uv run python -m signal_dataset

# 4. Output files
#   data/dataset.npz   — train/val/test splits (X, C, y per split)
#   data/signals_raw.npz — all clean and noisy signal vectors

# 5. (Optional) Generate visualizations
uv run python -m signal_dataset.visualize
```

---

## Project Structure

```
HW1/
├── src/signal_dataset/
│   ├── sdk/sdk.py               # Single entry point (DatasetSDK)
│   ├── services/
│   │   ├── signal_generator.py  # Clean sine wave generation
│   │   ├── noise_injector.py    # Gaussian + burst noise injection
│   │   ├── windower.py          # Sliding window + one-hot C vector
│   │   └── dataset_builder.py   # Splits, .npz serialization
│   ├── shared/
│   │   ├── config.py            # Config loading + validation
│   │   ├── gatekeeper.py        # API rate-limit gatekeeper
│   │   └── version.py           # Version tracking
│   ├── constants.py             # Math/physical constants
│   ├── main.py                  # CLI entry point
│   └── visualize.py             # Visualization script
├── tests/
│   ├── unit/                    # Unit tests (TDD: RED→GREEN→REFACTOR)
│   └── integration/             # End-to-end pipeline tests
├── config/
│   ├── setup.json               # All dataset parameters
│   └── rate_limits.json         # API gatekeeper rate limits
├── docs/
│   ├── PRD.md                   # Product Requirements Document
│   ├── PLAN.md                  # Architecture plan
│   ├── TODO.md                  # Task list
│   └── PRD_dataset_generation.md
├── results/                     # Generated visualizations
├── notebooks/                   # Jupyter analysis notebook
├── pyproject.toml               # All dependencies (uv)
└── uv.lock
```

---

## Configuration

All parameters are in `config/setup.json`:

| Key | Description | Default |
|-----|-------------|---------|
| `dataset.duration_sec` | Signal duration in seconds | 10 |
| `dataset.sample_rate_hz` | Sampling rate | 1000 |
| `dataset.window_size` | Context window size W | 10 |
| `dataset.train_ratio` | Train split fraction | 0.70 |
| `dataset.val_ratio` | Validation fraction | 0.15 |
| `dataset.test_ratio` | Test fraction | 0.15 |
| `dataset.random_seed` | Reproducibility seed | 42 |
| `noise.gaussian.sigma_amp` | Gaussian amplitude σ per signal | [0.05, 0.05, 0.03, 0.02] |
| `noise.burst.probability` | Burst event probability per sample | 0.01 |

---

## Testing

```bash
# Run all tests with coverage
uv run pytest tests/

# Run unit tests only
uv run pytest tests/unit/

# Run integration tests
uv run pytest tests/integration/
```

Coverage: **≥ 85%** (enforced — build fails below threshold).
Current: **97.79%**

---

## Linting

```bash
uv run ruff check
```

Zero errors enforced. Configuration in `pyproject.toml`.

---

## Design Rationale — Why These Four Signals?

The four sine waves were deliberately engineered to create a composite signal that is complex enough
to defeat fully-connected networks (FCNs) yet structured enough for Long Short-Term Memory (LSTM)
networks to succeed. Each component applies distinct pressure on a different architectural weakness.

### Signal Roles

| Signal | Role | Architecture Stress |
|--------|------|---------------------|
| **s1** — 5 Hz, A=2.0 | Dominant slow-moving trend | FCN cannot distinguish trend from noise without temporal context |
| **s2** — 15 Hz, A=1.5, φ=π/4 | Phase-shifted mid-band | Tests whether the network memorises absolute starting state vs. reacts to instantaneous slope — RNNs that rely on hidden state alone fail when phase is unknown |
| **s3** — 50 Hz, A=0.8 | High-frequency rapid variation | Exceeds the effective "memory horizon" of vanilla RNNs; vanishing gradients prevent back-propagation across the ~20 samples per cycle needed to track phase accurately |
| **s4** — 100 Hz, A=0.3 | Near-Nyquist structural oscillation | Only 10 samples per cycle at 1000 Hz; mimics high-frequency structural noise and forces the model to distinguish injected Gaussian/burst noise from a real signal component |
| **s5** | Composite (s1+s2+s3+s4) | Source of the context window X; the model must un-mix it given a one-hot conditioning vector C |

### Why FCNs Fail

An FCN maps a fixed-size input window to an output through purely spatial (pointwise) transformations.
It has no mechanism to track phase evolution across samples or accumulate evidence about frequency.
Given X of width W=10 (only 10 ms at 1000 Hz), a FCN cannot reliably separate the 5 Hz trend
from the 100 Hz component — both look like arbitrary amplitudes without temporal context.

### Why Vanilla RNNs Degrade

RNNs carry a hidden state across time, which helps with s1's slow trend. However:
- **s3 (50 Hz)**: requires tracking phase over ~20 samples per cycle. Vanishing gradients make
  this unreliable during back-propagation through time.
- **s2 (π/4 phase shift)**: the initial hidden state is zero, so the network must infer the
  absolute phase from context — difficult when the window is short and noise is present.
- **Burst noise** causes gradient spikes that destabilize RNN training without gating.

### Why LSTMs Succeed

The LSTM's cell state acts as a persistent, gated memory. The forget gate suppresses irrelevant
history; the input gate selectively writes new phase/amplitude information:
- The cell state can maintain a running estimate of frequency and phase across many time steps.
- The output gate exposes only the relevant part of memory to the current prediction.
- Combined with the one-hot conditioning vector C (appended to each window), the LSTM can
  specialise its cell dynamics per signal — essentially implementing conditional source separation.

### Conditional Source Separation Task

At inference time the model receives:

```
input = concat([x_w, C])   shape: (W + 5,) = (15,)
```

where `x_w` is a window from `s5_noisy` and `C` is a one-hot vector indicating which clean
signal component to recover. The label `y` is the scalar clean value of that component at the
window centre. This framing mimics real-world scenarios such as:

- Isolating a heartbeat frequency from a composite biomedical signal
- Extracting a carrier from a multi-tone RF signal
- Separating structural vibration modes in an accelerometer stream

The noise parameters (σ_amp, σ_phase, burst probability) are tunable via `config/setup.json`,
allowing sensitivity analysis of how each architecture degrades as the signal-to-noise ratio drops.

---

## Screenshots

### Clean Signal Components
![clean_signals](results/clean_signals.png)

### Clean vs Noisy Signals
![noisy_vs_clean](results/noisy_vs_clean.png)

### Frequency Spectrum (Clean vs Noisy)
![frequency_spectrum](results/frequency_spectrum.png)

### Noise Distribution Analysis
![noise_distribution](results/noise_distribution.png)

### Burst Noise Events on s1
![burst_events](results/burst_events.png)

### Dataset Split Distribution
![dataset_splits](results/dataset_splits.png)

### Context Window + One-Hot Selector Examples
![context_window_example](results/context_window_example.png)

---

## Next Phase

Phase 2 will train RNN, Fully Connected, and LSTM models on this dataset to learn
to isolate the selected clean signal component from the noisy composite.

---

## License

MIT License — see `LICENSE` for details.

---

## Author

Khaled Mnaa — khaled.mnaa43@gmail.com
