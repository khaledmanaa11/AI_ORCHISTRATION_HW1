"""Single source of truth for all physical and mathematical constants.

No business logic here — only constant definitions derived from math or domain invariants.
All signal parameters (amplitude, frequency, phase) come from config, not this file.
"""

import math

TWO_PI = 2.0 * math.pi

# Fallback defaults — config values take priority in all real code paths
DEFAULT_DURATION_SEC = 10
DEFAULT_SAMPLE_RATE_HZ = 1000
DEFAULT_WINDOW_SIZE = 10

N_SIGNALS = 4           # number of base sine components (s1–s4)
N_TOTAL_SIGNALS = 5     # including the composite s5

COMPOSITE_SIGNAL_ID = "s5"
SIGNAL_IDS = ["s1", "s2", "s3", "s4", "s5"]

N_SELECTOR_CLASSES = 5  # length of one-hot C vector (one per signal)

# Keys used when saving dataset.npz
DATASET_NPZ_KEY_X_TRAIN = "X_train"
DATASET_NPZ_KEY_X_VAL = "X_val"
DATASET_NPZ_KEY_X_TEST = "X_test"

DATASET_NPZ_KEY_C_TRAIN = "C_train"
DATASET_NPZ_KEY_C_VAL = "C_val"
DATASET_NPZ_KEY_C_TEST = "C_test"

DATASET_NPZ_KEY_Y_TRAIN = "y_train"
DATASET_NPZ_KEY_Y_VAL = "y_val"
DATASET_NPZ_KEY_Y_TEST = "y_test"

# Keys used when saving signals_raw.npz
RAW_NPZ_KEY_CLEAN = "clean_signals"
RAW_NPZ_KEY_NOISY = "noisy_signals"
RAW_NPZ_KEY_TIME = "time_axis"

NOISE_TYPE_GAUSSIAN = "gaussian"
NOISE_TYPE_BURST = "burst"
