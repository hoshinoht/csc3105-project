"""
preprocessing.py — Data cleaning, normalization, and train/test splitting.

This module implements the Data Preparation stage of the 3D analytics pipeline:
  1. Data Reduction: Drops constant-valued columns (CH, BITRATE, PRFR) that carry
     no discriminative information across the dataset.
  2. Data Transformation: Normalizes the 1016 CIR amplitude samples by dividing
     by RXPACC (received preamble count) to compensate for hardware accumulator
     variations across measurements.
  3. Feature Scaling: Applies StandardScaler (zero mean, unit variance) to the 11
     scalar features so that algorithms sensitive to feature magnitude (e.g.,
     Logistic Regression, neural networks) are not biased by differing scales.
  4. Train/Test Split: Stratified 80/20 split to preserve class balance (LOS/NLOS).

Libraries: numpy (array ops), pandas (DataFrame), sklearn (StandardScaler, train_test_split)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Columns that are identical across all 42,000 samples — no predictive value
CONSTANT_COLS = ['CH', 'BITRATE', 'PRFR']

# The 11 scalar features retained after dropping constants.
# These include hardware diagnostics (FP_AMP1-3, RXPACC, FRAME_LEN, PREAM_LEN),
# signal quality metrics (STDEV_NOISE, CIR_PWR, MAX_NOISE), and the measured
# time-of-flight range (RANGE) and first-path index (FP_IDX).
SCALAR_FEATURES = [
    'RANGE', 'FP_IDX', 'FP_AMP1', 'FP_AMP2', 'FP_AMP3',
    'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC',
    'FRAME_LEN', 'PREAM_LEN',
]

# Column names for the 1016 CIR time-domain samples (1 ns resolution each)
CIR_COLS = [f'CIR{i}' for i in range(1016)]


def preprocess(df):
    """
    Clean and normalize the raw dataset.

    Steps:
      1. Drop constant columns (CH, BITRATE, PRFR) — data reduction.
      2. Normalize CIR amplitudes by RXPACC — per-preamble-pulse normalization
         to remove accumulator-dependent gain variation.
      3. Check for NaN/Inf values introduced by normalization (e.g., RXPACC=0)
         and replace with 0.

    Parameters:
        df (pd.DataFrame): Raw dataset from load_dataset().

    Returns:
        pd.DataFrame: Cleaned and CIR-normalized dataset.
    """
    df = df.copy()

    # Step 1: Drop constant columns that have no variance across samples
    df.drop(columns=CONSTANT_COLS, inplace=True)
    print(f"Dropped constant columns: {CONSTANT_COLS}")

    # Step 2: Normalize CIR by RXPACC (per-preamble-pulse normalization)
    # The raw CIR amplitude is accumulated over RXPACC preamble symbols, so
    # dividing by RXPACC gives a per-pulse representation, making samples
    # comparable regardless of the number of accumulated preamble symbols.
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    rxpacc = df['RXPACC'].values[:, np.newaxis]  # shape: (N, 1) for broadcasting
    df[cir_cols] = df[cir_cols].values / rxpacc

    # Step 3: Handle potential NaN/Inf values from division (e.g., RXPACC=0)
    nan_count = df[cir_cols].isna().sum().sum()
    inf_count = np.isinf(df[cir_cols].values).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  Warning: {nan_count} NaN, {inf_count} Inf in CIR — replacing with 0")
        df[cir_cols] = df[cir_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Normalized {len(cir_cols)} CIR columns by RXPACC")
    return df


def scale_and_split(df, test_size=0.2, random_state=42):
    """
    Apply StandardScaler to scalar features and perform stratified train/test split.

    StandardScaler transforms each feature to zero mean and unit variance:
      z = (x - mean) / std
    This is essential for distance-based algorithms and neural networks.

    The split is stratified by the NLOS label to ensure both train and test sets
    maintain the original 50/50 LOS/NLOS class balance.

    Parameters:
        df (pd.DataFrame): Preprocessed dataset from preprocess().
        test_size (float): Fraction of data reserved for testing (default 0.2 = 80/20 split).
        random_state (int): Seed for reproducible splitting.

    Returns:
        df_scaled (pd.DataFrame): Dataset with scaled scalar features.
        train_idx (np.ndarray): Integer indices of training samples.
        test_idx (np.ndarray): Integer indices of test samples.
        scaler (StandardScaler): Fitted scaler for inverse transform or new data.
    """
    scaler = StandardScaler()
    df_scaled = df.copy()

    # Fit scaler on scalar features and transform in-place
    df_scaled[SCALAR_FEATURES] = scaler.fit_transform(df[SCALAR_FEATURES])

    # Stratified split: preserves LOS/NLOS ratio in both train and test sets
    y = df_scaled['NLOS'].values
    train_idx, test_idx = train_test_split(
        np.arange(len(df_scaled)), test_size=test_size,
        stratify=y, random_state=random_state,
    )
    print(f"Train/test split: {len(train_idx)} / {len(test_idx)} (stratified)")
    return df_scaled, train_idx, test_idx, scaler
