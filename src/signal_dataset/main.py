"""CLI entry point for the signal_dataset package.

Usage:
    uv run python -m signal_dataset [--config PATH] [--rate-limits PATH] [--verbose]
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="signal-dataset",
        description="Generate a noisy sine signal dataset for RNN/LSTM training.",
    )
    parser.add_argument(
        "--config",
        default="config/setup.json",
        help="Path to setup.json (default: config/setup.json)",
    )
    parser.add_argument(
        "--rate-limits",
        default="config/rate_limits.json",
        help="Path to rate_limits.json (default: config/rate_limits.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/",
        help="Output directory for generated files (default: data/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point: load config, generate dataset, print summary.

    Returns 0 on success, 1 on error.
    """
    args = parse_args()
    try:
        from signal_dataset.sdk.sdk import DatasetSDK

        if args.verbose:
            print(f"Loading config from: {args.config}")

        t0 = time.perf_counter()
        sdk = DatasetSDK(
            config_path=Path(args.config),
            rate_limits_path=Path(args.rate_limits),
        )
        result = sdk.generate_dataset()
        elapsed = time.perf_counter() - t0

        split = result.split_result
        print(f"Dataset saved  : {result.dataset_path}")
        print(f"Raw signals    : {result.raw_path}")
        print(f"X_train shape  : {split.X_train.shape}")
        print(f"X_val shape    : {split.X_val.shape}")
        print(f"X_test shape   : {split.X_test.shape}")
        print(f"Elapsed        : {elapsed:.2f}s")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
