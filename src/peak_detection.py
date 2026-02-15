"""Extract two dominant paths from each CIR measurement."""

import numpy as np
from scipy.signal import find_peaks


def extract_two_paths(df):
    """
    For each sample, find path 1 (near FP_IDX) and path 2 (next strongest).

    Returns arrays: path1_idx, path1_amp, path2_idx, path2_amp (all shape [N,]).
    """
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    cir_data = df[cir_cols].values  # (N, 1016)
    fp_idx_raw = df['FP_IDX'].values  # raw FP_IDX values

    n_samples = len(df)
    path1_idx = np.zeros(n_samples, dtype=int)
    path1_amp = np.zeros(n_samples)
    path2_idx = np.zeros(n_samples, dtype=int)
    path2_amp = np.zeros(n_samples)

    n_cir = cir_data.shape[1]

    for i in range(n_samples):
        cir = cir_data[i]
        # FP_IDX maps to CIR column: FP_IDX is at column index FP_IDX - 730 roughly
        # but the CIR starts at column 15 in the original CSV, so FP_IDX indexes
        # into the CIR array offset. The dataset FP_IDX values are ~737-751.
        # CIR0 corresponds to sample 0, FP_IDX points into the CIR array directly.
        fp = int(fp_idx_raw[i])

        # Path 1: search ±10 around FP_IDX for actual peak
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

        # Path 2: find peaks excluding path 1 neighborhood
        p1_amp = path1_amp[i]
        if p1_amp <= 0:
            continue

        peaks, properties = find_peaks(
            cir,
            height=0.1 * p1_amp,
            distance=15,
            prominence=0.05 * p1_amp,
        )

        if len(peaks) == 0:
            continue

        # Exclude peaks within ±20 of path 1
        mask = np.abs(peaks - path1_idx[i]) > 20
        peaks = peaks[mask]
        if len(peaks) == 0:
            continue

        # Select strongest remaining peak
        peak_amps = cir[peaks]
        best = np.argmax(peak_amps)
        path2_idx[i] = peaks[best]
        path2_amp[i] = peak_amps[best]

    n_found = (path2_amp > 0).sum()
    print(f"Peak detection: path2 found in {n_found}/{n_samples} samples "
          f"({100 * n_found / n_samples:.1f}%)")

    return path1_idx, path1_amp, path2_idx, path2_amp
