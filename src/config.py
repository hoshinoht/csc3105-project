"""
config.py — Central configuration constants for the UWB LOS/NLOS pipeline.

This module holds seeds, split ratios, and a handful of hyperparameters that
are referenced from more than one place in the codebase. The intent is to
provide a single authoritative source so that changing, for example, the
random seed only has to happen once.

Migration status (intentionally partial to keep report numbers stable):
    - main.py and dl_training.py import from here.
    - Per-module src/ files still hold their own random_state=42 constants.
      Migrating them all would require re-running the full pipeline and is
      deferred until after the current report submission.

When extending this file, add only values that are genuinely shared. Values
used in exactly one place belong next to that usage.
"""

# ── Reproducibility ───────────────────────────────────────────────
# Seed used by numpy, sklearn, and PyTorch. Keep this aligned with the
# random_state=42 constants currently hardcoded in the src/ modules.
RANDOM_STATE = 42

# Dedicated seed for the deep-learning pipeline. Currently the same as
# RANDOM_STATE; split out so it can be varied independently when doing
# DL-only reruns without disturbing the ML classifier / regressor numbers.
DL_SEED = 42

# ── Train/test split ──────────────────────────────────────────────
# Fraction of samples reserved for the test split in scale_and_split.
# Stratified by the NLOS label to preserve class balance.
TEST_SIZE = 0.2

# ── Synthetic data augmentation ───────────────────────────────────
# SMOTE target ratio: minority class is resampled to reach this fraction
# of the majority class count in the expanded two-path dataset.
SMOTE_TARGET_RATIO = 0.5

# CIR waveform augmentation for the CNN+Transformer pipeline.
CIR_AUG_FACTOR = 3
CIR_NOISE_LEVEL = 0.08
CIR_MAX_SHIFT = 4
CIR_SCALE_RANGE = (0.80, 1.20)

# ── SMOTE-retrained RF ────────────────────────────────────────────
# Hyperparameters for the Random Forest that is re-fit on the SMOTE-augmented
# training set inside the synthetic-data experiment block of main.py.
RF_SMOTE_N_ESTIMATORS = 300
