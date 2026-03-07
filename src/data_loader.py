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


def load_dataset(dataset_dir='dataset/'):
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
    frames = []

    # Iterate through sorted CSV files for deterministic loading order
    for f in sorted(os.listdir(dataset_dir)):
        if f.endswith('.csv'):
            frames.append(pd.read_csv(os.path.join(dataset_dir, f)))

    # Concatenate all environment CSVs into one DataFrame, reset row indices
    df = pd.concat(frames, ignore_index=True)

    # Print summary statistics for verification
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"  LOS: {(df['NLOS'] == 0).sum()}, NLOS: {(df['NLOS'] == 1).sum()}")
    return df
