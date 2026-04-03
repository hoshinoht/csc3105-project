"""
data_loader.py — Load and concatenate the 7 UWB indoor environment CSV files.

The UWB-LOS-NLOS dataset (Decawave DWM1000) contains measurements from 7 different
indoor environments (Office 1, Office 2, Small Apartment, Small Workshop, Kitchen with
Living Room, Bedroom, Boiler Room). Each environment contributes 3000 LOS and 3000 NLOS
samples, totalling 42,000 samples across all environments.

Each CSV row contains:
  - 15 scalar features (RANGE, FP_IDX, FP_AMP1-3, STDEV_NOISE, CIR_PWR, MAX_NOISE,
    RXPACC, CH, FRAME_LEN, PREAM_LEN, BITRATE, PRFR)
  - 1016 CIR amplitude samples (1 ns resolution channel impulse response)
  - 1 class label (NLOS: 1 = NLOS, 0 = LOS)

Library: pandas (DataFrame I/O), os (filesystem traversal)
"""

import os
import pandas as pd


def load_dataset(dataset_dir=None):
    """
    Load all CSV parts from the dataset directory and return a single DataFrame.

    The dataset is split across multiple CSV files (one per indoor environment).
    We sort by filename for reproducibility, then concatenate into a single
    DataFrame with shape (42000, 1031).

    Parameters:
        dataset_dir (str): Path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: Combined dataset with 42000 rows x 1031 columns
                      (15 scalar features + 1016 CIR samples + 1 NLOS label).
    """
    # Default to 'dataset/' relative to the project root (where main.py lives),
    # not relative to the caller's working directory.
    if dataset_dir is None:
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')

    frames = []

    # Iterate through sorted CSV files for deterministic loading order
    csv_files = sorted(f for f in os.listdir(dataset_dir) if f.endswith('.csv'))
    for i, f in enumerate(csv_files):
        part = pd.read_csv(os.path.join(dataset_dir, f))
        part['ENV_ID'] = i
        frames.append(part)

    # Concatenate all environment CSVs into one DataFrame, reset row indices.
    # copy() defragments the DataFrame (avoids PerformanceWarning from insert).
    df = pd.concat(frames, ignore_index=True).copy()

    # Print summary statistics for verification
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"  LOS: {(df['NLOS'] == 0).sum()}, NLOS: {(df['NLOS'] == 1).sum()}")
    print(f"  Environments: {df['ENV_ID'].nunique()} (ENV_ID 0-{df['ENV_ID'].max()})")
    return df
