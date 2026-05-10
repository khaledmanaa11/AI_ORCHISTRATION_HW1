"""signal_dataset — Noisy Sine Signal Dataset Generator package.

Public API: use DatasetSDK as the sole entry point for all dataset operations.
"""

from signal_dataset.sdk.sdk import DatasetSDK
from signal_dataset.shared.version import __version__

__all__ = ["DatasetSDK", "__version__"]
