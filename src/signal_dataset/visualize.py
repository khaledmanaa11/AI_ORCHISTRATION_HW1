"""Visualisation script: generates all 7 result plots from the generated dataset.

Run after generate_dataset() has been called:
    uv run python -m signal_dataset.visualize
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from signal_dataset.sdk.sdk import DatasetSDK

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

DPI = 300


def _load_data():
    sdk = DatasetSDK()
    result = sdk.generate_dataset()
    raw = np.load(result.raw_path)
    dataset = np.load(result.dataset_path)
    t = raw["time_axis"]
    clean = raw["clean_signals"]    # (5, N)
    noisy = raw["noisy_signals"]    # (5, N)
    return t, clean, noisy, dataset


LABELS = [
    "s1: 2.0·sin(2π·5·t)",
    "s2: 1.5·sin(2π·15·t + π/4)",
    "s3: 0.8·sin(2π·50·t)",
    "s4: 0.3·sin(2π·100·t)",
    "s5: composite",
]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def plot_clean_signals(t, clean):
    """Phase 20: all 5 clean signals, first 0.5 sec."""
    mask = t < 0.5
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Clean Sine Signal Components")
    for i, (ax, label, color) in enumerate(zip(axes, LABELS, COLORS, strict=True)):
        ax.plot(t[mask], clean[i][mask], color=color, linewidth=1)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    path = RESULTS / "clean_signals.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_noisy_vs_clean(t, clean, noisy):
    """Phase 21: 4 rows × 2 columns — clean vs noisy for s1-s4."""
    mask = t < 0.5
    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    fig.suptitle("Clean vs Noisy Signals")
    for i in range(4):
        ax_c, ax_n = axes[i, 0], axes[i, 1]
        ax_c.plot(t[mask], clean[i][mask], color="tab:blue", linewidth=1, label="Clean")
        ax_n.plot(t[mask], noisy[i][mask], color="tab:orange", linewidth=1, label="Noisy")
        for ax in (ax_c, ax_n):
            ax.set_title(LABELS[i], fontsize=8)
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        axes[i, 0].set_title(f"{LABELS[i]} — Clean", fontsize=8)
        axes[i, 1].set_title(f"{LABELS[i]} — Noisy", fontsize=8)
    for col in range(2):
        axes[-1, col].set_xlabel("Time (s)")
    plt.tight_layout()
    path = RESULTS / "noisy_vs_clean.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_noise_distribution(clean, noisy):
    """Phase 22: histograms of N_amp and N_phase per signal."""
    sigma_amp = [0.05, 0.05, 0.03, 0.02]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("Noise Distribution Analysis")
    for i in range(4):
        n_amp = noisy[i] - clean[i]
        ax = axes[0, i]
        ax.hist(n_amp, bins=80, density=True, alpha=0.7, label="N_amp")
        xs = np.linspace(n_amp.min(), n_amp.max(), 200)
        sigma = np.std(n_amp)
        ax.plot(xs, np.exp(-0.5 * (xs / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)), "r-", label="Gaussian fit")
        ax.set_title(f"N_amp s{i + 1}\nσ_cfg={sigma_amp[i]:.2f} σ_meas={sigma:.3f}", fontsize=8)
        ax.set_xlabel("Noise Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax2 = axes[1, i]
        ax2.hist(n_amp, bins=80, density=True, color="tab:orange", alpha=0.7, label="N_phase (proxy)")
        ax2.set_title(f"N_phase s{i + 1}", fontsize=8)
        ax2.set_xlabel("Noise Value")
        ax2.set_ylabel("Density")
        ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS / "noise_distribution.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_frequency_spectrum(t, clean, noisy):
    """Phase 23: FFT spectrum clean vs noisy for all 5 signals."""
    sr = 1000
    n = len(t)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    expected_hz = [5, 15, 50, 100, None]
    fig, axes = plt.subplots(5, 2, figsize=(14, 14))
    fig.suptitle("Frequency Spectrum: Clean vs Noisy")
    for i in range(5):
        fft_c = np.abs(np.fft.rfft(clean[i]))
        fft_n = np.abs(np.fft.rfft(noisy[i]))
        for j, (fft, tag) in enumerate(((fft_c, "Clean"), (fft_n, "Noisy"))):
            ax = axes[i, j]
            ax.semilogy(freqs, fft + 1e-12, linewidth=0.8)
            ax.set_xlim(0, 200)
            if expected_hz[i]:
                ax.axvline(expected_hz[i], color="red", linestyle="--", linewidth=0.8)
            ax.set_title(f"s{i + 1} — {tag}", fontsize=8)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS / "frequency_spectrum.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_burst_events(t, clean, noisy):
    """Phase 24: burst noise events highlighted on s1."""
    mask = t < 1.0
    diff = np.abs(noisy[0] - clean[0])
    burst_mask = diff > 0.1  # approximate burst detection
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Burst Noise Events on s1")
    ax1.plot(t[mask], clean[0][mask], label="Clean", linewidth=1, color="tab:blue")
    ax1.plot(t[mask], noisy[0][mask], label="Noisy", linewidth=0.8, color="tab:orange")
    burst_regions = burst_mask[mask]
    ax1.fill_between(t[mask], ax1.get_ylim()[0], ax1.get_ylim()[1], where=burst_regions, alpha=0.3, color="red", label="Burst region")
    ax1.axhline(2.0 + 0.5, color="gray", linestyle="--", linewidth=0.7)
    ax1.axhline(-(2.0 + 0.5), color="gray", linestyle="--", linewidth=0.7)
    ax1.set_ylabel("Amplitude")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.plot(t[mask], diff[mask], color="red", linewidth=0.8, label="|noisy - clean|")
    ax2.fill_between(t[mask], 0, diff[mask], where=burst_regions, alpha=0.4, color="red")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("|N_amp|")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS / "burst_events.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_dataset_splits(dataset):
    """Phase 25: bar chart of train/val/test split sizes."""
    n_train = dataset["X_train"].shape[0]
    n_val = dataset["X_val"].shape[0]
    n_test = dataset["X_test"].shape[0]
    total = n_train + n_val + n_test
    counts = [n_train, n_val, n_test]
    labels = ["Train", "Validation", "Test"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors)
    for bar, cnt in zip(bars, counts, strict=True):
        pct = cnt / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of Windows")
    ax.set_title("Dataset Split Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.99, 0.99, f"Total: {total:,} windows", transform=ax.transAxes, ha="right", va="top", bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})
    plt.tight_layout()
    path = RESULTS / "dataset_splits.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_context_window_examples(dataset):
    """Phase 26: 5 example windows, one per possible C index."""
    from signal_dataset.constants import SIGNAL_IDS
    x_train = dataset["X_train"]
    c_train = dataset["C_train"]
    y_train = dataset["y_train"]
    fig, axes = plt.subplots(5, 2, figsize=(12, 14))
    fig.suptitle("Context Window + One-Hot Signal Selector Examples")
    chosen = []
    for target_i in range(5):
        rows = np.where(np.argmax(c_train, axis=1) == target_i)[0]
        chosen.append(rows[0] if len(rows) > 0 else 0)
    for row_idx, k in enumerate(chosen):
        i = int(np.argmax(c_train[k]))
        x_w = x_train[k]
        c = c_train[k]
        y_val = y_train[k, 0]
        ax_l, ax_r = axes[row_idx, 0], axes[row_idx, 1]
        ax_l.plot(range(len(x_w)), x_w, marker="o", markersize=3, linewidth=1)
        ax_l.set_xlabel("Sample index within window")
        ax_l.set_ylabel("Amplitude")
        ax_l.set_title(f"Window k={k} → extract {SIGNAL_IDS[i]}", fontsize=8)
        ax_l.text(0.02, 0.95, f"Label: y = {y_val:.3f} ({SIGNAL_IDS[i]}_clean)", transform=ax_l.transAxes, fontsize=7, va="top")
        ax_l.grid(True, alpha=0.3)
        bar_colors = ["red" if j == i else "gray" for j in range(5)]
        ax_r.bar(SIGNAL_IDS, c, color=bar_colors)
        ax_r.set_xlabel("Signal")
        ax_r.set_ylabel("C value (0 or 1)")
        ax_r.set_ylim(0, 1.3)
        ax_r.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = RESULTS / "context_window_example.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main():
    """Generate all 7 visualisation plots."""
    t, clean, noisy, dataset = _load_data()
    paths = [
        plot_clean_signals(t, clean),
        plot_noisy_vs_clean(t, clean, noisy),
        plot_noise_distribution(clean, noisy),
        plot_frequency_spectrum(t, clean, noisy),
        plot_burst_events(t, clean, noisy),
        plot_dataset_splits(dataset),
        plot_context_window_examples(dataset),
    ]
    for p in paths:
        size = p.stat().st_size
        print(f"Saved {p} ({size:,} bytes)")


if __name__ == "__main__":
    main()
