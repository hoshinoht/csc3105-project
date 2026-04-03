"""
clustering.py — Unsupervised learning baseline using K-Means clustering.

This module provides an unsupervised learning perspective on the UWB CIR data,
complementing the supervised classification pipeline. It addresses the project
requirement for exploring clustering as a data mining technique.

The approach applies K-Means (k=2) to the same 18-feature vectors used by the
supervised classifiers, then measures how well the discovered clusters align
with the true LOS/NLOS labels. This serves two purposes:

1. **Baseline comparison**: Quantifies how much structure the LOS/NLOS classes
   have in feature space without any label information. If K-Means achieves
   reasonable agreement with the true labels, it confirms that the hand-crafted
   features capture meaningful physical differences between LOS and NLOS signals.

2. **Cluster quality analysis**: Silhouette scores measure how well-separated
   the clusters are. High silhouette scores (>0.5) indicate distinct, compact
   groups — validating the feature engineering decisions.

Additionally, PCA is used to project the 18-dimensional feature space down to
2D for visualisation, showing the cluster structure and how it relates to the
true labels.

Libraries: sklearn (KMeans, PCA, silhouette_score, adjusted_rand_score),
           numpy, matplotlib
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    confusion_matrix,
    accuracy_score,
)


def run_kmeans_analysis(X_train, y_train, X_test, y_test, random_state=42):
    """
    Run K-Means clustering (k=2) and compare clusters to true LOS/NLOS labels.

    K-Means partitions the feature space into k clusters by minimising within-
    cluster variance (sum of squared distances to each cluster centroid). With
    k=2, it discovers the two most natural groupings in the data — if these
    align with LOS/NLOS, it validates the feature space structure.

    Since cluster labels are arbitrary (cluster 0 could correspond to either
    LOS or NLOS), we try both label assignments and pick the one that maximises
    accuracy (equivalent to solving the optimal label permutation problem).

    Parameters:
        X_train (np.ndarray): Training feature matrix, shape (N_train, 18).
        y_train (np.ndarray): Training labels (0=LOS, 1=NLOS).
        X_test (np.ndarray): Test feature matrix, shape (N_test, 18).
        y_test (np.ndarray): Test labels (0=LOS, 1=NLOS).
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: {
            'model': fitted KMeans object,
            'train_labels': cluster assignments for training data,
            'test_labels': cluster assignments for test data,
            'train_accuracy': best-case accuracy on training set,
            'test_accuracy': best-case accuracy on test set,
            'silhouette_train': silhouette score on training data,
            'silhouette_test': silhouette score on test data,
            'ari': adjusted Rand index (label-permutation-invariant),
            'pca_model': fitted PCA for 2D visualisation,
            'X_test_2d': PCA-projected test features for plotting,
        }
    """
    print("\n--- K-Means Clustering (k=2) ---")

    # Fit K-Means on training data
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    train_clusters = kmeans.fit_predict(X_train)
    test_clusters = kmeans.predict(X_test)

    # Resolve cluster-to-label mapping: try both assignments, pick the best
    # Assignment A: cluster 0 → LOS (0), cluster 1 → NLOS (1)
    acc_a = accuracy_score(y_test, test_clusters)
    # Assignment B: cluster 0 → NLOS (1), cluster 1 → LOS (0)
    flipped = 1 - test_clusters
    acc_b = accuracy_score(y_test, flipped)

    if acc_b > acc_a:
        # Flip all assignments to the better mapping
        test_clusters = flipped
        train_clusters = 1 - train_clusters
        best_acc = acc_b
    else:
        best_acc = acc_a

    train_acc = accuracy_score(y_train, train_clusters)

    # Silhouette score: measures cluster separation quality [-1, 1]
    # +1 = perfectly separated, 0 = overlapping, -1 = misassigned
    # Use a subsample for efficiency on large datasets
    n_sil = min(10000, len(X_train))
    rng = np.random.RandomState(random_state)
    sil_idx = rng.choice(len(X_train), n_sil, replace=False)
    sil_train = silhouette_score(X_train[sil_idx], train_clusters[sil_idx])
    sil_test = silhouette_score(X_test, kmeans.predict(X_test))

    # Adjusted Rand Index: measures cluster-label agreement regardless of
    # label permutation. ARI = 1.0 means perfect agreement, 0 = random.
    ari = adjusted_rand_score(y_test, test_clusters)

    # PCA projection for 2D visualisation
    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(X_train)
    X_test_2d = pca.transform(X_test)

    print(f"  Train accuracy (best mapping): {train_acc:.4f}")
    print(f"  Test accuracy  (best mapping): {best_acc:.4f}")
    print(f"  Silhouette score (train): {sil_train:.4f}")
    print(f"  Silhouette score (test):  {sil_test:.4f}")
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  PCA explained variance: {pca.explained_variance_ratio_[:2].sum():.2%}")

    return {
        'model': kmeans,
        'train_labels': train_clusters,
        'test_labels': test_clusters,
        'train_accuracy': train_acc,
        'test_accuracy': best_acc,
        'silhouette_train': sil_train,
        'silhouette_test': sil_test,
        'ari': ari,
        'pca_model': pca,
        'X_test_2d': X_test_2d,
    }


def run_elbow_silhouette_analysis(X_train, k_range=range(2, 11), random_state=42):
    """
    Run K-Means for a range of k values and record inertia + silhouette scores.

    This produces the data needed for the Elbow Method (plotting inertia vs k
    to find the "elbow" where adding clusters yields diminishing returns) and
    the Silhouette Method (finding k that maximises average silhouette score).

    Silhouette computation uses a random subsample (max 10,000 samples) to
    keep runtime manageable since silhouette_score is O(n^2).

    Parameters:
        X_train (np.ndarray): Training feature matrix, shape (N, D).
        k_range (range): Range of k values to evaluate (default 2..10).
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: {
            'k_values': list of k values tested,
            'inertias': list of within-cluster sum of squares per k,
            'silhouettes': list of silhouette scores per k,
        }
    """
    print("\n--- Elbow / Silhouette Analysis ---")

    # Subsample for silhouette computation (O(n^2) complexity)
    n_sil = min(10000, len(X_train))
    rng = np.random.RandomState(random_state)
    sil_idx = rng.choice(len(X_train), n_sil, replace=False)
    X_sil = X_train[sil_idx]

    k_values = list(k_range)
    inertias = []
    silhouettes = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_sil)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_sil, labels)
        silhouettes.append(sil)

    # Print summary table
    print(f"  {'k':>3s}  {'Inertia':>12s}  {'Silhouette':>10s}")
    print(f"  {'---':>3s}  {'------------':>12s}  {'----------':>10s}")
    for k, inertia, sil in zip(k_values, inertias, silhouettes):
        print(f"  {k:3d}  {inertia:12.1f}  {sil:10.4f}")

    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouettes': silhouettes,
    }


def run_dbscan_analysis(X_train, X_test, y_test, random_state=42):
    """
    Run DBSCAN clustering and compare clusters to true LOS/NLOS labels.

    DBSCAN discovers clusters of arbitrary shape by grouping densely packed points,
    marking isolated points as noise (-1). Unlike K-Means, it does not require
    specifying k in advance.

    Parameters:
        X_train (np.ndarray): Training feature matrix, shape (N_train, 18).
        X_test (np.ndarray): Test feature matrix, shape (N_test, 18).
        y_test (np.ndarray): Test labels (0=LOS, 1=NLOS).
        random_state (int): Random seed for PCA reproducibility.

    Returns:
        dict: {test_labels, X_test_2d, n_clusters, n_noise, accuracy}
    """
    from sklearn.cluster import DBSCAN

    print("\n--- DBSCAN Clustering (eps=2.0, min_samples=10) ---")

    dbscan = DBSCAN(eps=2.0, min_samples=10)
    test_labels = dbscan.fit_predict(X_test)

    n_clusters = len(set(test_labels)) - (1 if -1 in test_labels else 0)
    n_noise = (test_labels == -1).sum()

    # Compute accuracy on non-noise samples only
    non_noise_mask = test_labels != -1
    if non_noise_mask.sum() > 0 and n_clusters >= 2:
        y_nn = y_test[non_noise_mask]
        cl_nn = test_labels[non_noise_mask]
        # Try both label mappings
        acc_a = accuracy_score(y_nn, cl_nn)
        acc_b = accuracy_score(y_nn, 1 - cl_nn)
        accuracy = max(acc_a, acc_b)
    else:
        accuracy = 0.0

    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(X_train)
    X_test_2d = pca.transform(X_test)

    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise} ({100*n_noise/len(test_labels):.1f}%)")
    print(f"  Accuracy (non-noise only): {accuracy:.4f}")

    return {
        'test_labels': test_labels,
        'X_test_2d': X_test_2d,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'accuracy': accuracy,
    }
