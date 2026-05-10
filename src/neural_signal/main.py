"""CLI entry point — delegates entirely to NeuralSignalSDK."""

from __future__ import annotations

from pathlib import Path

from neural_signal.sdk.sdk import NeuralSignalSDK


def main() -> None:
    """Train FCN, RNN, and LSTM models; save all results and plots."""
    root = Path(__file__).resolve().parent.parent.parent
    sdk = NeuralSignalSDK(
        config_path=root / "config" / "setup.json",
        rate_limits_path=root / "config" / "rate_limits.json",
    )
    results = sdk.run_all()
    for name, res in results.items():
        print(f"{name.upper():5s}  test_mse={res.test_mse:.6f}  epochs={res.epochs_trained}")


if __name__ == "__main__":
    main()
