"""Load and concatenate the 7 UWB dataset CSV files."""

import os
import pandas as pd


def load_dataset(dataset_dir='dataset/'):
    """Load all CSV parts and return a single DataFrame (42000 x 1031)."""
    frames = []
    for f in sorted(os.listdir(dataset_dir)):
        if f.endswith('.csv'):
            frames.append(pd.read_csv(os.path.join(dataset_dir, f)))
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"  LOS: {(df['NLOS'] == 0).sum()}, NLOS: {(df['NLOS'] == 1).sum()}")
    return df
