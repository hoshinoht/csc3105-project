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
         FP_IDX (first path index) for the actual local maximum. This accounts for
         slight offsets between the hardware estimate and the true peak.
      2. Path 2: Use scipy.signal.find_peaks with adaptive thresholds relative to
         Path 1's amplitude (height ≥ 10%, prominence ≥ 5% of Path 1). Exclude
         peaks within ±20 samples of Path 1 to avoid detecting sidelobes. Select
         the strongest remaining peak as Path 2.

    Parameters:
        df (pd.DataFrame): Preprocessed dataset containing CIR columns and FP_IDX.

    Returns:
        path1_idx (np.ndarray): CIR sample index of Path 1 for each measurement.
        path1_amp (np.ndarray): Amplitude of Path 1 at the detected index.
        path2_idx (np.ndarray): CIR sample index of Path 2 (0 if not found).
        path2_amp (np.ndarray): Amplitude of Path 2 (0.0 if not found).
    """
    # Extract CIR columns (exclude CIR_PWR which is a summary statistic, not a sample)
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    cir_data = df[cir_cols].values  # shape: (N, 1016)
    fp_idx_raw = df['FP_IDX'].values  # hardware-reported first path index

    n_samples = len(df)
    path1_idx = np.zeros(n_samples, dtype=int)
    path1_amp = np.zeros(n_samples)
    path2_idx = np.zeros(n_samples, dtype=int)
    path2_amp = np.zeros(n_samples)

    n_cir = cir_data.shape[1]  # 1016 time-domain samples

    for i in range(n_samples):
        cir = cir_data[i]
        fp = int(fp_idx_raw[i])  # hardware first-path index

        # ── Path 1: Refine FP_IDX to actual peak ──────────────────────
        # The hardware FP_IDX can be slightly offset from the true amplitude
        # peak. Search a ±10-sample window around FP_IDX for the maximum.
        lo = max(0, fp - 10)
        hi = min(n_cir, fp + 11)
        window = cir[lo:hi]
        if len(window) > 0:
            local_peak = np.argmax(window)
            path1_idx[i] = lo + local_peak  # convert local index to global CIR index
            path1_amp[i] = window[local_peak]
        else:
            # Fallback: use FP_IDX directly if window is empty (edge case)
            path1_idx[i] = fp
            path1_amp[i] = cir[min(fp, n_cir - 1)]

        # ── Path 2: Find the next strongest peak ─────────────────────
        p1_amp = path1_amp[i]
        if p1_amp <= 0:
            continue  # skip if Path 1 has no valid amplitude

        # Use scipy's find_peaks with thresholds relative to Path 1's amplitude:
        #   - height ≥ 10% of Path 1 amplitude (ignore noise-level peaks)
        #   - distance ≥ 15 samples between peaks (avoid sidelobe clusters)
        #   - prominence ≥ 5% of Path 1 amplitude (must be a true local maximum)
        peaks, properties = find_peaks(
            cir,
            height=0.1 * p1_amp,
            distance=15,
            prominence=0.05 * p1_amp,
        )

        if len(peaks) == 0:
            continue

        # Exclude peaks within ±20 samples of Path 1 to avoid sidelobes
        # or the same wavelet being counted twice
        mask = np.abs(peaks - path1_idx[i]) > 20
        peaks = peaks[mask]
        if len(peaks) == 0:
            continue

        # Select the strongest remaining peak as Path 2
        peak_amps = cir[peaks]
        best = np.argmax(peak_amps)
        path2_idx[i] = peaks[best]
        path2_amp[i] = peak_amps[best]

    # Report detection rate — not all CIRs have a clearly separable second path
    n_found = (path2_amp > 0).sum()
    print(f"Peak detection: path2 found in {n_found}/{n_samples} samples "
          f"({100 * n_found / n_samples:.1f}%)")

    return path1_idx, path1_amp, path2_idx, path2_amp
