"""
peak_detection.py — Extract the two dominant propagation paths from each CIR measurement.

In UWB indoor positioning, the Channel Impulse Response (CIR) captures the multipath
propagation profile between anchor and tag. The key insight from the problem statement
is that each CIR contains (at least) two dominant paths:
  - Path 1: The first arriving path, near the FP_IDX (first path index).
             If the anchor-tag pair has line-of-sight, this is the direct (LOS) path.
  - Path 2: The next strongest peak in the CIR, representing either a reflected
             (NLOS) signal or a secondary multipath component.

The classification task requires identifying whether these paths are LOS or NLOS:
  - If Path 1 is LOS → Path 2 is NLOS (reflection/multipath)
  - If Path 1 is NLOS → Path 2 is also NLOS (both are obstructed)

Libraries: numpy (array operations), scipy.signal.find_peaks (peak detection)
"""

import numpy as np
from scipy.signal import find_peaks


def extract_two_paths(df):
    """
    For each of the 42,000 CIR samples, detect the two strongest propagation paths.

    Algorithm:
      1. Path 1: Search within a ±10-sample window around the hardware-reported
         FP_IDX (first path index) for the actual local maximum.
      2. Path 2: Use lower adaptive threshold (5% of Path 1 vs original 10%)
         and rank candidates by prominence instead of raw amplitude.
         Exclude peaks within ±20 samples of Path 1.

    Parameters:
        df (pd.DataFrame): Preprocessed dataset containing CIR columns and FP_IDX.

    Returns:
        path1_idx (np.ndarray): CIR sample index of Path 1 for each measurement.
        path1_amp (np.ndarray): Amplitude of Path 1 at the detected index.
        path2_idx (np.ndarray): CIR sample index of Path 2 (0 if not found).
        path2_amp (np.ndarray): Amplitude of Path 2 (0.0 if not found).
    """
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    cir_data = df[cir_cols].values  # shape: (N, 1016)
    fp_idx_raw = df['FP_IDX'].values

    n_samples = len(df)
    path1_idx = np.zeros(n_samples, dtype=int)
    path1_amp = np.zeros(n_samples)
    path2_idx = np.zeros(n_samples, dtype=int)
    path2_amp = np.zeros(n_samples)

    n_cir = cir_data.shape[1]  # 1016 time-domain samples

    for i in range(n_samples):
        cir = cir_data[i]
        fp = int(fp_idx_raw[i])

        # ── Path 1: Refine FP_IDX to actual peak ──────────────────────
        lo = max(0, fp - 10)
        hi = min(n_cir, fp + 11)
        window = cir[lo:hi]
        if len(window) > 0:
            local_peak = np.argmax(window)
            path1_idx[i] = lo + local_peak
            path1_amp[i] = window[local_peak]
        else:
            path1_idx[i] = fp
            path1_amp[i] = cir[min(fp, n_cir - 1)]

        # ── Path 2: Find the next strongest peak ─────────────────────
        p1_amp = path1_amp[i]
        if p1_amp <= 0:
            continue

        # Apply 3-sample moving average smoothing for more robust peak detection
        kernel = np.ones(3) / 3.0
        cir_smooth = np.convolve(cir, kernel, mode='same')

        # Adaptive threshold: 2% of p1 amplitude to detect weaker secondary paths
        peaks, properties = find_peaks(
            cir_smooth,
            height=0.02 * p1_amp,
            distance=15,
            prominence=0.02 * p1_amp,
        )

        if len(peaks) == 0:
            continue

        # Exclude peaks within ±15 samples of Path 1
        mask = np.abs(peaks - path1_idx[i]) > 15
        peaks = peaks[mask]
        prominences = properties['prominences'][mask] if 'prominences' in properties else None
        if len(peaks) == 0:
            continue

        # Rank by prominence (from find_peaks) instead of raw amplitude
        if prominences is not None and len(prominences) > 0:
            best = np.argmax(prominences)
        else:
            best = np.argmax(cir[peaks])
        path2_idx[i] = peaks[best]
        # Read amplitude from original (unsmoothed) CIR
        path2_amp[i] = cir[peaks[best]]

    # Report detection rate — not all CIRs have a clearly separable second path
    n_found = (path2_amp > 0).sum()
    print(f"Peak detection: path2 found in {n_found}/{n_samples} samples "
          f"({100 * n_found / n_samples:.1f}%)")

    return path1_idx, path1_amp, path2_idx, path2_amp
