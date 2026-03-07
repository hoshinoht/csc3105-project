"""
synthetic_data.py — Synthetic data generation for performance robustness testing.

This module addresses Data Preparation requirement VI: exploring whether synthetic
data improves model robustness. It implements two complementary augmentation
strategies for UWB CIR signals:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   Generates synthetic feature-space samples by interpolating between existing
   samples and their k-nearest neighbours. Applied to the hand-crafted 18-feature
   vectors to balance or augment the training set.

2. **CIR Waveform Augmentation**:
   Generates synthetic CIR waveforms by applying physically-motivated perturbations
   to real signals:
     - Gaussian noise injection (simulates receiver thermal noise)
     - Temporal jitter (simulates clock synchronisation imprecision)
     - Amplitude scaling (simulates path loss variation across distances)
   These augmented waveforms are suitable for training the CNN+Transformer DL model.

The module also provides an evaluation function that compares model performance
with and without synthetic data, measuring robustness improvement.

Libraries: numpy, sklearn (SMOTE via imblearn if available, otherwise manual),
           scipy.ndimage (for temporal shift)
"""

import numpy as np
from scipy.ndimage import shift as ndimage_shift


# ══════════════════════════════════════════════════════════════════════
# Strategy 1: SMOTE for Feature-Space Augmentation
# ══════════════════════════════════════════════════════════════════════

def apply_smote(X_train, y_train, target_ratio=1.0, k_neighbors=5, random_state=42):
    """
    Apply SMOTE to generate synthetic feature-space samples for the minority class.

    SMOTE works by selecting a minority-class sample, finding its k nearest
    neighbours in feature space, and creating new samples along the line
    segments connecting them. This produces realistic synthetic samples that
    lie within the convex hull of the minority class.

    In our two-path expanded dataset, LOS is the minority class (25%) and
    NLOS is the majority (75%). SMOTE can balance this to improve LOS recall.

    Parameters:
        X_train (np.ndarray): Training feature matrix, shape (N, 18).
        y_train (np.ndarray): Training labels (0=LOS, 1=NLOS).
        target_ratio (float): Desired minority/majority ratio after SMOTE.
                              1.0 = fully balanced, 0.5 = half balanced.
        k_neighbors (int): Number of nearest neighbours for interpolation.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_resampled (np.ndarray): Augmented feature matrix with synthetic samples.
        y_resampled (np.ndarray): Augmented labels.
        n_synthetic (int): Number of synthetic samples generated.
    """
    rng = np.random.RandomState(random_state)

    # Identify minority and majority classes
    classes, counts = np.unique(y_train, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_count = counts.max()
    minority_count = counts.min()

    # Calculate how many synthetic samples to generate
    target_count = int(majority_count * target_ratio)
    n_synthetic = max(0, target_count - minority_count)

    if n_synthetic == 0:
        print("  SMOTE: No synthetic samples needed (classes already balanced)")
        return X_train, y_train, 0

    # Extract minority class samples
    minority_mask = y_train == minority_class
    X_minority = X_train[minority_mask]

    # Compute pairwise distances for k-NN (Euclidean)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X_minority)))
    nn.fit(X_minority)
    distances, indices = nn.kneighbors(X_minority)

    # Generate synthetic samples via interpolation
    synthetic_samples = np.zeros((n_synthetic, X_train.shape[1]))
    for i in range(n_synthetic):
        # Pick a random minority sample
        idx = rng.randint(0, len(X_minority))
        # Pick a random neighbour (skip index 0 which is the sample itself)
        neighbor_idx = indices[idx, rng.randint(1, indices.shape[1])]
        # Interpolate: new_sample = sample + lambda * (neighbour - sample)
        lam = rng.uniform(0, 1)
        synthetic_samples[i] = X_minority[idx] + lam * (X_minority[neighbor_idx] - X_minority[idx])

    # Combine original data with synthetic samples
    synthetic_labels = np.full(n_synthetic, minority_class)
    X_resampled = np.vstack([X_train, synthetic_samples])
    y_resampled = np.concatenate([y_train, synthetic_labels])

    print(f"  SMOTE: Generated {n_synthetic} synthetic {['LOS', 'NLOS'][minority_class]} samples")
    print(f"  Class distribution: LOS={int((y_resampled == 0).sum())}, "
          f"NLOS={int((y_resampled == 1).sum())}")

    return X_resampled, y_resampled, n_synthetic


# ══════════════════════════════════════════════════════════════════════
# Strategy 2: CIR Waveform Augmentation for Deep Learning
# ══════════════════════════════════════════════════════════════════════

def augment_cir_noise(cir_data, noise_level=0.05, random_state=42):
    """
    Add Gaussian noise to CIR waveforms to simulate receiver thermal noise.

    In real UWB systems, the CIR is contaminated by additive white Gaussian
    noise (AWGN) from the receiver front-end. Augmenting with noise at
    different levels trains the model to be robust to varying SNR conditions.

    Parameters:
        cir_data (np.ndarray): CIR waveforms, shape (N, 1016).
        noise_level (float): Noise standard deviation as fraction of each
                             sample's CIR standard deviation (default 0.05 = 5%).
        random_state (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Augmented CIR waveforms with added Gaussian noise.
    """
    rng = np.random.RandomState(random_state)
    augmented = cir_data.copy()

    # Scale noise relative to each sample's own signal magnitude
    sample_stds = np.std(cir_data, axis=1, keepdims=True)
    # Avoid zero division for flat signals
    sample_stds = np.maximum(sample_stds, 1e-8)

    noise = rng.randn(*cir_data.shape) * sample_stds * noise_level
    augmented += noise

    return augmented


def augment_cir_jitter(cir_data, max_shift=3, random_state=42):
    """
    Apply random temporal jitter (shift) to CIR waveforms.

    Clock synchronisation imprecision between UWB anchor and tag can cause
    small temporal offsets in the CIR. Augmenting with random shifts trains
    the model to be invariant to slight timing errors.

    Parameters:
        cir_data (np.ndarray): CIR waveforms, shape (N, 1016).
        max_shift (int): Maximum shift in samples (±max_shift), default 3.
        random_state (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Temporally jittered CIR waveforms.
    """
    rng = np.random.RandomState(random_state)
    augmented = np.zeros_like(cir_data)

    for i in range(len(cir_data)):
        # Random integer shift in range [-max_shift, +max_shift]
        shift_amount = rng.randint(-max_shift, max_shift + 1)
        # scipy.ndimage.shift with constant boundary (zero-pad edges)
        augmented[i] = ndimage_shift(cir_data[i], shift_amount, mode='constant', cval=0.0)

    return augmented


def augment_cir_amplitude(cir_data, scale_range=(0.8, 1.2), random_state=42):
    """
    Apply random amplitude scaling to CIR waveforms.

    Path loss varies with distance, obstruction, and antenna orientation,
    causing overall CIR amplitude variations. Augmenting with random scaling
    trains the model to focus on waveform shape rather than absolute magnitude.

    Parameters:
        cir_data (np.ndarray): CIR waveforms, shape (N, 1016).
        scale_range (tuple): Min and max scaling factors (default 0.8 to 1.2).
        random_state (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Amplitude-scaled CIR waveforms.
    """
    rng = np.random.RandomState(random_state)

    # Random scale factor per sample (uniform distribution)
    scales = rng.uniform(scale_range[0], scale_range[1], size=(len(cir_data), 1))
    augmented = cir_data * scales

    return augmented


def generate_augmented_cir(cir_data, scalars, labels, augmentation_factor=1,
                           noise_level=0.05, max_shift=2, scale_range=(0.85, 1.15),
                           random_state=42):
    """
    Generate a full set of augmented CIR training data using all three strategies.

    For each original sample, creates `augmentation_factor` synthetic samples by
    applying combined noise injection + temporal jitter + amplitude scaling.

    Parameters:
        cir_data (np.ndarray): Original CIR waveforms, shape (N, 1016).
        scalars (np.ndarray): Original scalar features, shape (N, 11).
        labels (np.ndarray): Original labels, shape (N,).
        augmentation_factor (int): Number of augmented copies per original sample.
        noise_level (float): Gaussian noise level for augmentation.
        max_shift (int): Maximum temporal jitter in samples.
        scale_range (tuple): Amplitude scaling range.
        random_state (int): Random seed for reproducibility.

    Returns:
        cir_aug (np.ndarray): Combined original + augmented CIR data.
        scalars_aug (np.ndarray): Combined original + repeated scalar data.
        labels_aug (np.ndarray): Combined original + repeated labels.
        n_synthetic (int): Number of synthetic samples generated.
    """
    rng = np.random.RandomState(random_state)
    n_original = len(cir_data)
    n_synthetic = n_original * augmentation_factor

    print(f"\n  CIR Augmentation: generating {n_synthetic} synthetic samples "
          f"({augmentation_factor}x factor)")
    print(f"    Noise level: {noise_level}, Max jitter: +/-{max_shift} samples, "
          f"Amplitude scale: {scale_range}")

    all_aug_cir = []
    all_aug_scalars = []
    all_aug_labels = []

    for aug_round in range(augmentation_factor):
        seed = random_state + aug_round  # Different seed per augmentation round

        # Apply all three augmentation strategies in sequence
        aug_cir = augment_cir_noise(cir_data, noise_level=noise_level, random_state=seed)
        aug_cir = augment_cir_jitter(aug_cir, max_shift=max_shift, random_state=seed + 1000)
        aug_cir = augment_cir_amplitude(aug_cir, scale_range=scale_range, random_state=seed + 2000)

        all_aug_cir.append(aug_cir)
        all_aug_scalars.append(scalars.copy())  # Scalar features stay the same
        all_aug_labels.append(labels.copy())     # Labels stay the same

    # Combine original + all augmented rounds
    cir_aug = np.vstack([cir_data] + all_aug_cir)
    scalars_aug = np.vstack([scalars] + all_aug_scalars)
    labels_aug = np.concatenate([labels] + all_aug_labels)

    print(f"  Total training set: {len(cir_aug)} samples "
          f"({n_original} original + {n_synthetic} synthetic)")

    return cir_aug, scalars_aug, labels_aug, n_synthetic


# ══════════════════════════════════════════════════════════════════════
# Evaluation: Compare performance with and without synthetic data
# ══════════════════════════════════════════════════════════════════════

def evaluate_synthetic_impact(cls_results_original, cls_results_augmented, model_name):
    """
    Compare classification metrics before and after synthetic data augmentation.

    Prints a side-by-side comparison of accuracy, AUC, and any improvement.

    Parameters:
        cls_results_original (dict): Results without synthetic data.
        cls_results_augmented (dict): Results with synthetic data.
        model_name (str): Name of the model being compared.
    """
    orig = cls_results_original[model_name]
    aug = cls_results_augmented[model_name]

    print(f"\n{'='*50}")
    print(f"Synthetic Data Impact: {model_name}")
    print(f"{'='*50}")
    print(f"  {'Metric':<15} {'Original':>10} {'Augmented':>10} {'Delta':>10}")
    print(f"  {'-'*45}")

    for metric in ['accuracy', 'auc']:
        o = orig[metric]
        a = aug[metric]
        delta = a - o
        sign = '+' if delta >= 0 else ''
        print(f"  {metric:<15} {o:>10.4f} {a:>10.4f} {sign}{delta:>9.4f}")
