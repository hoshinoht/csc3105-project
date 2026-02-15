"""Clean, normalize, and split the dataset."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CONSTANT_COLS = ['CH', 'BITRATE', 'PRFR']
SCALAR_FEATURES = [
    'RANGE', 'FP_IDX', 'FP_AMP1', 'FP_AMP2', 'FP_AMP3',
    'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC',
    'FRAME_LEN', 'PREAM_LEN',
]
CIR_COLS = [f'CIR{i}' for i in range(1016)]  # CIR0..CIR1015


def preprocess(df):
    """Normalize CIR, drop constants, and return processed DataFrame."""
    df = df.copy()

    # Drop constant columns
    df.drop(columns=CONSTANT_COLS, inplace=True)
    print(f"Dropped constant columns: {CONSTANT_COLS}")

    # Normalize CIR by RXPACC (per-preamble-pulse normalization)
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    rxpacc = df['RXPACC'].values[:, np.newaxis]
    df[cir_cols] = df[cir_cols].values / rxpacc

    # Check for NaN/Inf
    nan_count = df[cir_cols].isna().sum().sum()
    inf_count = np.isinf(df[cir_cols].values).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  Warning: {nan_count} NaN, {inf_count} Inf in CIR — replacing with 0")
        df[cir_cols] = df[cir_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Normalized {len(cir_cols)} CIR columns by RXPACC")
    return df


def scale_and_split(df, test_size=0.2, random_state=42):
    """StandardScale scalar features and do stratified train/test split."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[SCALAR_FEATURES] = scaler.fit_transform(df[SCALAR_FEATURES])

    y = df_scaled['NLOS'].values
    train_idx, test_idx = train_test_split(
        np.arange(len(df_scaled)), test_size=test_size,
        stratify=y, random_state=random_state,
    )
    print(f"Train/test split: {len(train_idx)} / {len(test_idx)} (stratified)")
    return df_scaled, train_idx, test_idx, scaler
