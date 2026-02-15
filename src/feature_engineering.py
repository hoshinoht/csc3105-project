"""Build per-path features, two-path labels, and distance labels."""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


SHARED_SCALAR = [
    'FP_AMP1', 'FP_AMP2', 'FP_AMP3', 'STDEV_NOISE',
    'CIR_PWR', 'MAX_NOISE', 'RXPACC', 'FRAME_LEN', 'PREAM_LEN',
]

SPEED_OF_LIGHT_NS = 0.2998  # meters per nanosecond


def _compute_path_features(cir_data, path_idx, path_amp, other_amp, stdev_noise):
    """Compute per-path features from CIR data around the given peak."""
    n = len(path_idx)
    n_cir = cir_data.shape[1]
    window_half = 15

    features = {
        'path_idx': path_idx.astype(float),
        'path_amp': path_amp,
        'rise_time': np.zeros(n),
        'decay_time': np.zeros(n),
        'kurtosis_local': np.zeros(n),
        'energy_ratio': np.zeros(n),
        'peak_to_noise': np.zeros(n),
        'amplitude_ratio': np.zeros(n),
    }

    total_energy = np.sum(cir_data ** 2, axis=1)

    for i in range(n):
        idx = int(path_idx[i])
        amp = path_amp[i]

        if amp <= 0:
            continue

        lo = max(0, idx - window_half)
        hi = min(n_cir, idx + window_half + 1)
        window = cir_data[i, lo:hi]
        local_idx = idx - lo  # position of peak within window

        # Rise time: samples from 10% to peak going left
        threshold = 0.1 * amp
        rise = 0
        for j in range(local_idx, -1, -1):
            if window[j] < threshold:
                break
            rise += 1
        features['rise_time'][i] = rise

        # Decay time: samples from peak to 10% going right
        decay = 0
        for j in range(local_idx, len(window)):
            if window[j] < threshold:
                break
            decay += 1
        features['decay_time'][i] = decay

        # Kurtosis of local window
        if len(window) > 3:
            features['kurtosis_local'][i] = kurtosis(window, fisher=True)

        # Energy ratio
        local_energy = np.sum(window ** 2)
        if total_energy[i] > 0:
            features['energy_ratio'][i] = local_energy / total_energy[i]

        # Peak to noise
        if stdev_noise[i] > 0:
            features['peak_to_noise'][i] = amp / stdev_noise[i]

        # Amplitude ratio
        if other_amp[i] > 0:
            features['amplitude_ratio'][i] = amp / other_amp[i]

    return pd.DataFrame(features)


def build_features(df, path1_idx, path1_amp, path2_idx, path2_amp):
    """
    Build the two-path expanded dataset (84k rows from 42k samples).

    Returns:
        features_df: DataFrame with per-path + shared scalar features
        labels_cls: NLOS labels (0=LOS, 1=NLOS) for two-path classification
        labels_range: distance labels for each path
        path_ids: array indicating which path (1 or 2)
    """
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    cir_data = df[cir_cols].values
    stdev_noise = df['STDEV_NOISE'].values
    original_nlos = df['NLOS'].values
    original_range = df['RANGE'].values
    fp_idx = df['FP_IDX'].values

    # Per-path features
    p1_feats = _compute_path_features(cir_data, path1_idx, path1_amp, path2_amp, stdev_noise)
    p2_feats = _compute_path_features(cir_data, path2_idx, path2_amp, path1_amp, stdev_noise)

    p1_feats.columns = [f'p_{c}' for c in p1_feats.columns]
    p2_feats.columns = [f'p_{c}' for c in p2_feats.columns]

    # Shared scalar features
    shared = df[SHARED_SCALAR].reset_index(drop=True)

    # Build path 1 rows
    path1_df = pd.concat([p1_feats, shared], axis=1)
    path1_df['path_id'] = 1

    # Build path 2 rows
    path2_df = pd.concat([p2_feats, shared], axis=1)
    path2_df['path_id'] = 2

    # Two-path labeling
    # LOS sample (NLOS=0): path1=LOS(0), path2=NLOS(1)
    # NLOS sample (NLOS=1): path1=NLOS(1), path2=NLOS(1)
    labels_p1 = original_nlos.copy()  # same as original
    labels_p2 = np.ones(len(df), dtype=int)  # always NLOS

    # Distance labels
    range_p1 = original_range.copy()
    range_p2 = original_range + (path2_idx - fp_idx) * SPEED_OF_LIGHT_NS

    # Stack path1 and path2
    features_df = pd.concat([path1_df, path2_df], ignore_index=True)
    labels_cls = np.concatenate([labels_p1, labels_p2])
    labels_range = np.concatenate([range_p1, range_p2])
    path_ids = np.concatenate([np.ones(len(df)), 2 * np.ones(len(df))])

    # Handle edge cases: where path2 was not found, set features to indicate
    no_path2 = path2_amp == 0
    n_no_path2 = no_path2.sum()
    if n_no_path2 > 0:
        print(f"  {n_no_path2} samples had no second path detected")

    print(f"Feature engineering: {len(features_df)} rows "
          f"(LOS={int((labels_cls == 0).sum())}, NLOS={int((labels_cls == 1).sum())})")

    return features_df, labels_cls, labels_range, path_ids
