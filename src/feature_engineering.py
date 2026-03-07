"""
feature_engineering.py — Build per-path features, two-path labels, and distance labels.

This module performs feature extraction and transformation (Data Preparation stage):
  1. Extracts 8 per-path features from the CIR waveform around each detected peak:
     - path_idx: CIR sample index of the detected peak
     - path_amp: Amplitude at the peak
     - rise_time: Number of samples from 10% amplitude threshold to peak (left side)
     - decay_time: Number of samples from peak to 10% threshold (right side)
     - kurtosis_local: Statistical kurtosis of the local ±15-sample window
     - energy_ratio: Fraction of total CIR energy contained in the local window
     - peak_to_noise: Signal-to-noise ratio (peak amplitude / noise std deviation)
     - amplitude_ratio: Ratio of this path's amplitude to the other path's amplitude

  2. Appends 9 shared scalar features from the Decawave hardware diagnostics.

  3. Adds physically meaningful shared features:
     - path_delay_separation: CIR sample delay between path 2 and path 1
     - path1_to_fp_offset: deviation of detected path 1 from hardware FP_IDX
     - relative_power: first-path power fraction (FP_AMP1 / CIR_PWR)

  4. Performs two-path label expansion (42K → 84K samples):
     - For LOS samples: Path 1 = LOS, Path 2 = NLOS
     - For NLOS samples: Path 1 = NLOS, Path 2 = NLOS
     This follows the physical reasoning that LOS is always the shortest path.

  5. Computes distance labels for range estimation:
     - Path 1 distance = original RANGE measurement
     - Path 2 distance = RANGE + (path2_idx - FP_IDX) × speed_of_light

Libraries: numpy (array ops), pandas (DataFrame construction), scipy.stats.kurtosis
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis


# Shared scalar features from the Decawave DWM1000 hardware diagnostics.
# These are the same for both paths since they describe the overall signal,
# not individual propagation paths.
SHARED_SCALAR = [
    'FP_AMP1', 'FP_AMP2', 'FP_AMP3',  # Three first-path amplitude components
    'STDEV_NOISE',                        # Standard deviation of channel noise
    'CIR_PWR',                            # Total channel impulse response power
    'MAX_NOISE',                          # Maximum noise level in the CIR
    'RXPACC',                             # Received preamble symbol count
    'FRAME_LEN',                          # Length of the transmitted frame
    'PREAM_LEN',                          # Preamble length
]

# Speed of light in meters per nanosecond — used to convert CIR sample offsets
# (1 ns resolution) into physical distances for Path 2 range estimation.
SPEED_OF_LIGHT_NS = 0.2998  # m/ns


def _compute_cir_stats(cir_data, path_idx):
    """
    Compute 5 CIR statistical features per path (vectorized for speed).

    Features capture the power delay profile shape and energy distribution,
    which differ between LOS (concentrated energy) and NLOS (spread energy).

    Returns:
        dict of 5 feature arrays, each shape (N,).
    """
    n = len(path_idx)
    n_cir = cir_data.shape[1]

    # Power delay profile: squared CIR amplitudes
    pdp = cir_data ** 2  # (N, 1016)
    total_power = pdp.sum(axis=1)  # (N,)
    valid = total_power > 0

    # Time index array for moment calculations
    k = np.arange(n_cir, dtype=float)  # (1016,)

    # Mean excess delay: first moment of PDP (vectorized)
    mean_delay = np.zeros(n)
    mean_delay[valid] = (pdp[valid] @ k) / total_power[valid]

    # RMS delay spread: sqrt of second central moment
    rms_delay = np.zeros(n)
    k_centered = k[np.newaxis, :] - mean_delay[:, np.newaxis]  # (N, 1016)
    variance = np.zeros(n)
    variance[valid] = np.sum(k_centered[valid] ** 2 * pdp[valid], axis=1) / total_power[valid]
    rms_delay[valid] = np.sqrt(variance[valid])

    # Kurtosis of full CIR (vectorized via scipy)
    kurtosis_full = scipy_kurtosis(cir_data, axis=1, fisher=True)

    # Max-to-mean ratio
    max_to_mean = np.zeros(n)
    cir_mean = cir_data.mean(axis=1)
    cir_max = cir_data.max(axis=1)
    pos_mean = cir_mean > 0
    max_to_mean[pos_mean] = cir_max[pos_mean] / cir_mean[pos_mean]

    # Tail energy ratio: energy from path_idx onward / total (vectorized)
    # Cumulative sum from the right: cumsum_r[i, j] = sum(pdp[i, j:])
    tail_energy_ratio = np.zeros(n)
    idx_arr = np.clip(path_idx.astype(int), 0, n_cir - 1)
    pdp_cumsum_r = np.cumsum(pdp[:, ::-1], axis=1)[:, ::-1]  # reverse cumsum
    tail_energy_ratio[valid] = pdp_cumsum_r[np.arange(n)[valid], idx_arr[valid]] / total_power[valid]

    return {
        'rms_delay_spread': rms_delay,
        'mean_excess_delay': mean_delay,
        'kurtosis_full': kurtosis_full,
        'max_to_mean': max_to_mean,
        'tail_energy_ratio': tail_energy_ratio,
    }


def _compute_path_features(cir_data, path_idx, path_amp, other_amp, stdev_noise):
    """
    Compute 8 hand-crafted features for one set of detected paths.

    These features capture the shape, strength, and quality of each propagation
    path relative to the noise floor and the other detected path. They encode
    domain knowledge about how LOS and NLOS signals differ in their CIR profiles:
    - LOS paths tend to have sharp rise/decay, high kurtosis, high SNR
    - NLOS paths tend to be broader, lower amplitude, more spread energy

    Parameters:
        cir_data (np.ndarray): CIR amplitudes, shape (N, 1016).
        path_idx (np.ndarray): Peak indices for this path, shape (N,).
        path_amp (np.ndarray): Peak amplitudes for this path, shape (N,).
        other_amp (np.ndarray): Peak amplitudes of the OTHER path, shape (N,).
        stdev_noise (np.ndarray): Noise standard deviation per sample, shape (N,).

    Returns:
        pd.DataFrame: 8-column DataFrame with per-path features.
    """
    n = len(path_idx)
    n_cir = cir_data.shape[1]
    window_half = 15  # Extract features from a ±15-sample window around the peak

    # Initialise feature arrays
    features = {
        'path_idx': path_idx.astype(float),  # Peak position in CIR (temporal)
        'path_amp': path_amp,                 # Peak amplitude
        'rise_time': np.zeros(n),             # Rising edge width (samples)
        'decay_time': np.zeros(n),            # Falling edge width (samples)
        'kurtosis_local': np.zeros(n),        # Peakedness of local waveform
        'energy_ratio': np.zeros(n),          # Local/total energy ratio
        'peak_to_noise': np.zeros(n),         # Signal-to-noise ratio
        'amplitude_ratio': np.zeros(n),       # This path vs other path amplitude
    }

    # Pre-compute total CIR energy for each sample (denominator for energy_ratio)
    total_energy = np.sum(cir_data ** 2, axis=1)

    for i in range(n):
        idx = int(path_idx[i])
        amp = path_amp[i]

        if amp <= 0:
            continue  # Skip samples where this path was not detected

        # Extract local window around the peak
        lo = max(0, idx - window_half)
        hi = min(n_cir, idx + window_half + 1)
        window = cir_data[i, lo:hi]
        local_idx = idx - lo  # Position of peak within the extracted window

        # ── Rise time: count samples from peak going LEFT until amplitude
        #    drops below 10% of peak. LOS signals have shorter rise times. ──
        threshold = 0.1 * amp
        rise = 0
        for j in range(local_idx, -1, -1):
            if window[j] < threshold:
                break
            rise += 1
        features['rise_time'][i] = rise

        # ── Decay time: count samples from peak going RIGHT until amplitude
        #    drops below 10%. NLOS signals tend to have longer decay tails. ──
        decay = 0
        for j in range(local_idx, len(window)):
            if window[j] < threshold:
                break
            decay += 1
        features['decay_time'][i] = decay

        # ── Local kurtosis: measures "peakedness" of the waveform shape.
        #    High kurtosis → sharp/spiky peak (typical of LOS direct paths).
        #    Low kurtosis → flat/spread shape (typical of NLOS scattered paths). ──
        if len(window) > 3:
            features['kurtosis_local'][i] = scipy_kurtosis(window, fisher=True)

        # ── Energy ratio: fraction of total CIR energy in this path's window.
        #    High ratio means this path dominates the CIR (likely LOS). ──
        local_energy = np.sum(window ** 2)
        if total_energy[i] > 0:
            features['energy_ratio'][i] = local_energy / total_energy[i]

        # ── Peak-to-noise ratio: path amplitude relative to channel noise.
        #    Higher SNR generally indicates a cleaner, more direct signal path. ──
        if stdev_noise[i] > 0:
            features['peak_to_noise'][i] = amp / stdev_noise[i]

        # ── Amplitude ratio: strength of this path vs the other detected path.
        #    Values > 1 mean this path is stronger than the other one. ──
        if other_amp[i] > 0:
            features['amplitude_ratio'][i] = amp / other_amp[i]

    # Merge CIR statistical features
    cir_stats = _compute_cir_stats(cir_data, path_idx)
    features.update(cir_stats)

    return pd.DataFrame(features)


def build_features(df, path1_idx, path1_amp, path2_idx, path2_amp):
    """
    Build the two-path expanded dataset (84K rows from 42K original samples).

    Each original sample produces TWO rows — one per detected path — with:
      - 13 per-path features (prefixed 'p_'): 8 waveform + 5 CIR statistics
      - 9 shared scalar features (from hardware diagnostics)
      - 3 shared derived features: path_delay_separation, path1_to_fp_offset, relative_power
    Total: 25 features per row.

    Two-path NLOS labeling logic (from the problem statement):
      - LOS sample (NLOS=0): Path 1 → LOS (0), Path 2 → NLOS (1)
      - NLOS sample (NLOS=1): Path 1 → NLOS (1), Path 2 → NLOS (1)
    Rationale: LOS is always the shortest/first-arriving path if it exists.

    Distance estimation labels:
      - Path 1 range = measured RANGE (time-of-flight distance)
      - Path 2 range = RANGE + (path2_idx - FP_IDX) × 0.2998 m/ns
        (converts CIR sample offset to physical distance)

    Parameters:
        df (pd.DataFrame): Preprocessed dataset.
        path1_idx, path1_amp: Path 1 detection results from extract_two_paths().
        path2_idx, path2_amp: Path 2 detection results from extract_two_paths().

    Returns:
        features_df (pd.DataFrame): 84K-row feature matrix (18 columns).
        labels_cls (np.ndarray): Binary NLOS labels for classification (84K,).
        labels_range (np.ndarray): Distance labels in metres for regression (84K,).
        path_ids (np.ndarray): Path identifier array (1 or 2) for each row (84K,).
    """
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    cir_data = df[cir_cols].values
    stdev_noise = df['STDEV_NOISE'].values
    original_nlos = df['NLOS'].values
    original_range = df['RANGE'].values
    fp_idx = df['FP_IDX'].values

    # ── Compute per-path features for each detected path ──────────────
    # Path 1 features: computed around path1 peak, with path2 as the "other" path
    p1_feats = _compute_path_features(cir_data, path1_idx, path1_amp, path2_amp, stdev_noise)
    # Path 2 features: computed around path2 peak, with path1 as the "other" path
    p2_feats = _compute_path_features(cir_data, path2_idx, path2_amp, path1_amp, stdev_noise)

    # Prefix column names with 'p_' to indicate these are per-path features
    p1_feats.columns = [f'p_{c}' for c in p1_feats.columns]
    p2_feats.columns = [f'p_{c}' for c in p2_feats.columns]

    # ── Append shared scalar features (same for both paths of a sample) ──
    shared = df[SHARED_SCALAR].reset_index(drop=True)

    # ── Compute physically meaningful shared features ───────────────
    # Path delay separation: temporal gap between the two detected paths
    path_delay_separation = path2_idx - path1_idx
    # Path 1 to FP offset: deviation of detected path 1 from hardware first-path index
    path1_to_fp_offset = path1_idx - fp_idx
    # Relative power: first-path power as fraction of total CIR power (strong LOS/NLOS discriminator)
    cir_pwr = df['CIR_PWR'].values
    relative_power = np.zeros(len(df))
    valid_pwr = cir_pwr > 0
    relative_power[valid_pwr] = df['FP_AMP1'].values[valid_pwr] / cir_pwr[valid_pwr]

    # Build Path 1 rows: per-path features + shared scalars + new shared features
    path1_df = pd.concat([p1_feats, shared], axis=1)
    path1_df['path_delay_separation'] = path_delay_separation
    path1_df['path1_to_fp_offset'] = path1_to_fp_offset.astype(float)
    path1_df['relative_power'] = relative_power

    # Build Path 2 rows: per-path features + shared scalars + new shared features
    path2_df = pd.concat([p2_feats, shared], axis=1)
    path2_df['path_delay_separation'] = path_delay_separation
    path2_df['path1_to_fp_offset'] = path1_to_fp_offset.astype(float)
    path2_df['relative_power'] = relative_power

    # ── Two-path NLOS labeling ────────────────────────────────────────
    # Path 1 inherits the original label: LOS(0) stays LOS, NLOS(1) stays NLOS
    labels_p1 = original_nlos.copy()
    # Path 2 is ALWAYS NLOS(1): even in LOS samples, the reflected path is NLOS
    labels_p2 = np.ones(len(df), dtype=int)

    # ── Distance labels for range estimation ──────────────────────────
    # Path 1: use the original measured range directly
    range_p1 = original_range.copy()
    # Path 2: original range + time offset converted to metres
    # The offset (path2_idx - FP_IDX) represents additional propagation time
    # in nanoseconds, multiplied by speed of light to get distance in metres
    range_p2 = original_range + (path2_idx - fp_idx) * SPEED_OF_LIGHT_NS

    # ── Stack Path 1 and Path 2 rows into expanded dataset ───────────
    # Rows 0..N-1 are Path 1, rows N..2N-1 are Path 2
    features_df = pd.concat([path1_df, path2_df], ignore_index=True)
    labels_cls = np.concatenate([labels_p1, labels_p2])
    labels_range = np.concatenate([range_p1, range_p2])
    path_ids = np.concatenate([np.ones(len(df)), 2 * np.ones(len(df))])

    # Report samples where Path 2 was not detected (amplitude = 0)
    no_path2 = path2_amp == 0
    n_no_path2 = no_path2.sum()
    if n_no_path2 > 0:
        print(f"  {n_no_path2} samples had no second path detected")

    print(f"Feature engineering: {len(features_df)} rows "
          f"(LOS={int((labels_cls == 0).sum())}, NLOS={int((labels_cls == 1).sum())})")

    return features_df, labels_cls, labels_range, path_ids
