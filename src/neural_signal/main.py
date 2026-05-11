"""CLI entry point — delegates entirely to NeuralSignalSDK."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_signal.sdk.sdk import NeuralSignalSDK


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(description="Train and evaluate neural signal denoising models.")
    p.add_argument(
        "--config", type=Path, default=root / "config" / "setup.json",
        help="Path to setup.json (default: config/setup.json)",
    )
    p.add_argument(
        "--rate-limits", type=Path, default=root / "config" / "rate_limits.json",
        help="Path to rate_limits.json (default: config/rate_limits.json)",
    )
    p.add_argument(
        "--model", choices=["fcn", "rnn", "lstm", "all"], default="all",
        help="Which model to train (default: all)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-epoch progress",
    )
    return p.parse_args()


def main() -> None:
    """Parse CLI args, run the requested model(s), and print results."""
    args = _parse_args()
    sdk = NeuralSignalSDK(
        config_path=args.config,
        rate_limits_path=args.rate_limits,
    )
    if args.model == "all":
        results = sdk.run_all()
    else:
        sdk.train_model(args.model)
        results = sdk.evaluate_all()

    for name, res in results.items():
        if args.model in ("all", name):
            print(f"{name.upper():5s}  test_mse={res.test_mse:.6f}  epochs={res.epochs_trained}")


if __name__ == "__main__":
    main()
