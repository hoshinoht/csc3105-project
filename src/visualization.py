"""
visualization.py — All visualization functions organized by 3D analytics stages.

This module implements the Data Visualization stage, generating plots that
illustrate the data characteristics, model performance, and analysis results.

Plots are organized by the three analytics stages:
  1. Data Preparation plots: class distribution, feature distributions,
     correlation heatmap, CIR examples, FP_IDX distribution.
  2. Data Mining plots: feature importance, confusion matrices, ROC curves,
     model comparison bar chart.
  3. Results/Analysis plots: predicted vs actual scatter, residual distributions,
     annotated CIR examples, transformer attention maps.

All plots are saved to the 'plots/' directory at 150 DPI for report inclusion.

Libraries: matplotlib (plotting), seaborn (heatmaps), numpy, pandas,
           sklearn.metrics (ConfusionMatrixDisplay)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


# Output directory for all generated plot images
PLOT_DIR = 'plots/'


def _savefig(name):
    """Save the current matplotlib figure to the plots directory and close it."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# Data Preparation Plots
# ══════════════════════════════════════════════════════════════════════

def plot_class_distribution(original_nlos, expanded_labels):
    """
    Bar chart of class distribution before and after two-path expansion.

    Shows how the dataset grows from 42K balanced samples (50/50 LOS/NLOS)
    to 84K imbalanced samples (25/75 LOS/NLOS) after the two-path expansion.
    This visualises the class labelling decision (Data Preparation step III).

    Parameters:
        original_nlos (np.ndarray): NLOS labels from the original 42K dataset.
        expanded_labels (np.ndarray): NLOS labels from the 84K expanded dataset.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: Original 42K dataset (balanced 50/50)
    vals, counts = np.unique(original_nlos, return_counts=True)
    axes[0].bar(['LOS', 'NLOS'], counts, color=['steelblue', 'coral'])
    axes[0].set_title('Original Dataset')
    axes[0].set_ylabel('Count')
    for j, c in enumerate(counts):
        axes[0].text(j, c + 200, str(c), ha='center')

    # Right panel: Expanded 84K dataset (imbalanced 25/75)
    vals2, counts2 = np.unique(expanded_labels, return_counts=True)
    labels2 = ['LOS' if v == 0 else 'NLOS' for v in vals2]
    axes[1].bar(labels2, counts2, color=['steelblue', 'coral'])
    axes[1].set_title('Two-Path Expanded')
    axes[1].set_ylabel('Count')
    for j, c in enumerate(counts2):
        axes[1].text(j, c + 500, str(c), ha='center')

    fig.suptitle('Class Distribution')
    _savefig('01_class_distribution.png')


def plot_feature_distributions(df):
    """
    Histograms/KDE of key scalar features by LOS/NLOS class.

    Visualises the feature distributions to understand which features show
    separation between LOS and NLOS classes — informing feature importance
    and the rationale for feature selection (Data Preparation step V).

    Parameters:
        df (pd.DataFrame): Raw dataset with scalar features and NLOS label.
    """
    feats = ['RANGE', 'FP_AMP1', 'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.ravel(), feats):
        for label, color in [(0, 'steelblue'), (1, 'coral')]:
            subset = df[df['NLOS'] == label][feat]
            ax.hist(subset, bins=50, alpha=0.5, color=color,
                    label='LOS' if label == 0 else 'NLOS', density=True)
        ax.set_title(feat)
        ax.legend(fontsize=8)
    fig.suptitle('Feature Distributions by LOS/NLOS')
    _savefig('02_feature_distributions.png')


def plot_correlation_heatmap(df):
    """
    Correlation heatmap of scalar features and the NLOS label.

    Reveals linear relationships between features, helping identify:
    - Redundant features (high inter-feature correlation)
    - Predictive features (high correlation with NLOS label)
    This informs data reduction decisions (Data Preparation step I).

    Parameters:
        df (pd.DataFrame): Raw dataset with scalar features and NLOS label.
    """
    scalar = ['RANGE', 'FP_IDX', 'FP_AMP1', 'FP_AMP2', 'FP_AMP3',
              'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC',
              'FRAME_LEN', 'PREAM_LEN', 'NLOS']
    corr = df[scalar].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Scalar Feature Correlation')
    _savefig('03_correlation_heatmap.png')


def plot_cir_examples(df, path1_idx, path1_amp, path2_idx, path2_amp, n=4):
    """
    Example CIR waveforms for LOS vs NLOS with detected path peaks marked.

    Provides visual understanding of the raw CIR data and validates the
    peak detection algorithm by overlaying detected Path 1 (blue) and
    Path 2 (red) positions on sample waveforms.

    Parameters:
        df (pd.DataFrame): Preprocessed dataset with CIR columns.
        path1_idx, path1_amp: Path 1 detection results.
        path2_idx, path2_amp: Path 2 detection results.
        n (int): Number of example samples to show per class (default 4).
    """
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))

    for row, (label, title) in enumerate([(0, 'LOS'), (1, 'NLOS')]):
        idxs = np.where(df['NLOS'].values == label)[0][:n]
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            cir = df.iloc[i][cir_cols].values.astype(float)
            ax.plot(cir, linewidth=0.5, color='gray')
            # Mark Path 1 (blue triangle, dashed line)
            ax.axvline(path1_idx[i], color='blue', linestyle='--', alpha=0.7, label='Path 1')
            ax.plot(path1_idx[i], path1_amp[i], 'bv', markersize=8)
            # Mark Path 2 if detected (red triangle, dashed line)
            if path2_amp[i] > 0:
                ax.axvline(path2_idx[i], color='red', linestyle='--', alpha=0.7, label='Path 2')
                ax.plot(path2_idx[i], path2_amp[i], 'r^', markersize=8)
            ax.set_title(f'{title} #{col + 1}')
            # Zoom to the region around the detected paths
            ax.set_xlim(max(0, path1_idx[i] - 50), min(len(cir), path1_idx[i] + 100))
            if col == 0:
                ax.set_ylabel('Amplitude')
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle('CIR Examples with Detected Peaks')
    _savefig('04_cir_examples.png')


def plot_fp_idx_distribution(df):
    """
    FP_IDX (First Path Index) distribution by LOS/NLOS class.

    Shows whether the hardware-reported first path index differs between
    LOS and NLOS conditions, which informs feature engineering decisions.

    Parameters:
        df (pd.DataFrame): Raw dataset with FP_IDX and NLOS columns.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [(0, 'steelblue'), (1, 'coral')]:
        subset = df[df['NLOS'] == label]['FP_IDX']
        ax.hist(subset, bins=50, alpha=0.5, color=color,
                label='LOS' if label == 0 else 'NLOS', density=True)
    ax.set_xlabel('FP_IDX')
    ax.set_ylabel('Density')
    ax.set_title('First Path Index Distribution')
    ax.legend()
    _savefig('05_fp_idx_distribution.png')


# ══════════════════════════════════════════════════════════════════════
# Data Mining Plots
# ══════════════════════════════════════════════════════════════════════

def plot_feature_importance(importances, feature_names, top_n=20):
    """
    Horizontal bar chart of Random Forest Gini feature importances.

    Ranks features by their contribution to classification accuracy,
    addressing Data Preparation requirement V (feature importance ranking).

    Parameters:
        importances (np.ndarray): Gini importance scores from Random Forest.
        feature_names (list): Names of the features corresponding to scores.
        top_n (int): Number of top features to display (default 20).
    """
    top_n = min(top_n, len(importances))
    idx = np.argsort(importances)[::-1][:top_n]  # Sort by importance descending
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[idx][::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)')
    _savefig('06_feature_importance.png')


def plot_confusion_matrices(cls_results, y_test):
    """
    Confusion matrix heatmaps for all classifiers side by side.

    Uses stored confusion matrices to support models with different test set
    sizes (ML models use 16,800 expanded samples, DL uses 8,400 original).

    Parameters:
        cls_results (dict): Classification results from all models.
        y_test (np.ndarray): Test labels (used for axis labels only).
    """
    n_models = len(cls_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, cls_results.items()):
        # Use pre-computed confusion matrix from each model's results
        cm = res['confusion_matrix']
        ConfusionMatrixDisplay(cm, display_labels=['LOS', 'NLOS']).plot(
            cmap='Blues', ax=ax,
        )
        ax.set_title(name)
    fig.suptitle('Confusion Matrices')
    _savefig('07_confusion_matrices.png')


def plot_roc_curves(cls_results):
    """
    ROC curves for all classifiers overlaid on the same axes.

    The ROC curve plots True Positive Rate vs False Positive Rate at
    varying classification thresholds. AUC (Area Under Curve) summarises
    the overall ranking quality -- higher AUC = better discrimination.

    Parameters:
        cls_results (dict): Classification results containing 'fpr', 'tpr', 'auc'.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in cls_results.items():
        ax.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Random classifier baseline
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    _savefig('08_roc_curves.png')


def plot_model_comparison(cls_results):
    """
    Grouped bar chart comparing accuracy and AUC across all models.

    Provides a quick visual summary of model performance for the
    Data Analysis stage.

    Parameters:
        cls_results (dict): Classification results containing 'accuracy' and 'auc'.
    """
    names = list(cls_results.keys())
    accs = [cls_results[n]['accuracy'] for n in names]
    aucs = [cls_results[n]['auc'] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accs, width, label='Accuracy', color='steelblue')
    ax.bar(x + width / 2, aucs, width, label='AUC', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.legend()
    _savefig('09_model_comparison.png')


# ══════════════════════════════════════════════════════════════════════
# Results / Analysis Plots
# ══════════════════════════════════════════════════════════════════════

def plot_predicted_vs_actual(reg_results_p1, y_test_p1, reg_results_p2, y_test_p2):
    """
    Predicted vs actual scatter plots for the best regressor on each path.

    Points along the y=x diagonal indicate perfect predictions. Deviation
    from the diagonal shows estimation error. The best model (lowest RMSE)
    is automatically selected for each path.

    Parameters:
        reg_results_p1 (dict): Regression results for Path 1.
        y_test_p1 (np.ndarray): True distances for Path 1 test set.
        reg_results_p2 (dict): Regression results for Path 2.
        y_test_p2 (np.ndarray): True distances for Path 2 test set.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, y_test, title in [
        (axes[0], reg_results_p1, y_test_p1, 'Path 1 Range'),
        (axes[1], reg_results_p2, y_test_p2, 'Path 2 Range'),
    ]:
        # Automatically select model with lowest RMSE
        best_name = min(results, key=lambda n: results[n]['rmse'])
        y_pred = results[best_name]['y_pred']
        ax.scatter(y_test, y_pred, alpha=0.1, s=5, color='steelblue')
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5)  # Perfect prediction line
        ax.set_xlabel('Actual (m)')
        ax.set_ylabel('Predicted (m)')
        ax.set_title(f'{title} -- {best_name}\n'
                      f'RMSE={results[best_name]["rmse"]:.3f}m')
    fig.suptitle('Predicted vs Actual Range')
    _savefig('10_predicted_vs_actual.png')


def plot_residuals(reg_results_p1, y_test_p1, reg_results_p2, y_test_p2):
    """
    Residual (error) distribution histograms for range estimation.

    Residual = actual - predicted. A well-calibrated model should produce
    residuals centred at zero with small spread. Heavy tails indicate
    outlier predictions. The red dashed line marks zero residual.

    Parameters:
        reg_results_p1 (dict): Regression results for Path 1.
        y_test_p1 (np.ndarray): True distances for Path 1 test set.
        reg_results_p2 (dict): Regression results for Path 2.
        y_test_p2 (np.ndarray): True distances for Path 2 test set.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, y_test, title in [
        (axes[0], reg_results_p1, y_test_p1, 'Path 1'),
        (axes[1], reg_results_p2, y_test_p2, 'Path 2'),
    ]:
        best_name = min(results, key=lambda n: results[n]['rmse'])
        residuals = y_test - results[best_name]['y_pred']
        ax.hist(residuals, bins=80, color='steelblue', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--')  # Zero-error reference
        ax.set_xlabel('Residual (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'{title} Residuals -- {best_name}')
    fig.suptitle('Residual Distributions')
    _savefig('11_residuals.png')


def plot_attention_map(dl_model, df, train_idx, test_idx, n=4):
    """
    Visualize transformer self-attention weights on sample CIR waveforms.

    For each sample, shows:
      - Top row: Raw CIR waveform coloured by true class (blue=LOS, red=NLOS)
      - Bottom row: Attention-weighted CIR showing which temporal regions
        the transformer focuses on for its LOS/NLOS decision.

    Attention weights are upsampled from the 127-step transformer sequence
    back to the original 1016-sample CIR resolution via linear interpolation.

    Parameters:
        dl_model: Trained CIRTransformerClassifier in mode for inference.
        df (pd.DataFrame): Scaled dataset with CIR columns and NLOS labels.
        train_idx (np.ndarray): Training sample indices (not used, kept for API).
        test_idx (np.ndarray): Test sample indices for selecting examples.
        n (int): Total number of examples to plot (split evenly LOS/NLOS).
    """
    import torch
    from src.preprocessing import SCALAR_FEATURES

    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    device = next(dl_model.parameters()).device
    dl_model.eval()

    # Select n/2 LOS and n/2 NLOS samples from the test set
    test_labels = df['NLOS'].values[test_idx]
    los_idxs = test_idx[test_labels == 0][:n // 2]
    nlos_idxs = test_idx[test_labels == 1][:n // 2]
    sample_idxs = np.concatenate([los_idxs, nlos_idxs])

    fig, axes = plt.subplots(2, len(sample_idxs), figsize=(5 * len(sample_idxs), 8))

    for col, idx in enumerate(sample_idxs):
        # Prepare single-sample tensors for forward pass
        cir_raw = df.iloc[idx][cir_cols].values.astype(float)
        cir_tensor = torch.tensor(cir_raw, dtype=torch.float32).unsqueeze(0).to(device)
        scalar_tensor = torch.tensor(
            df.iloc[idx][SCALAR_FEATURES].values.astype(float),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        # Forward pass to get prediction and trigger attention hook
        with torch.no_grad():
            logit = dl_model(cir_tensor, scalar_tensor)
            pred = 'NLOS' if logit.item() > 0 else 'LOS'

        true_label = 'NLOS' if df.iloc[idx]['NLOS'] == 1 else 'LOS'

        # Extract attention weights: [1, seq_len, seq_len] -> average over queries -> [seq_len]
        attn = dl_model.last_attention_weights
        if attn is not None:
            attn_avg = attn[0].mean(dim=0).cpu().numpy()  # Average across query positions
            # Upsample attention from 127 transformer steps to 1016 CIR samples
            attn_upsampled = np.interp(
                np.linspace(0, 1, len(cir_raw)),
                np.linspace(0, 1, len(attn_avg)),
                attn_avg,
            )
        else:
            attn_upsampled = np.zeros(len(cir_raw))

        # Top row: CIR waveform coloured by true class
        ax_top = axes[0, col]
        ax_top.plot(cir_raw, color='gray', linewidth=0.5)
        ax_top.fill_between(
            range(len(cir_raw)),
            0, cir_raw,
            alpha=0.3,
            color='red' if true_label == 'NLOS' else 'blue',
        )
        ax_top.set_title(f'True: {true_label}, Pred: {pred}')
        if col == 0:
            ax_top.set_ylabel('CIR Amplitude')

        # Bottom row: attention-weighted overlay (orange = high attention)
        ax_bot = axes[1, col]
        ax_bot.plot(cir_raw, color='gray', linewidth=0.5, alpha=0.5)
        ax_bot.fill_between(
            range(len(cir_raw)),
            0,
            attn_upsampled * cir_raw.max(),
            alpha=0.6,
            color='orange',
            label='Attention',
        )
        ax_bot.set_xlabel('CIR Sample Index')
        if col == 0:
            ax_bot.set_ylabel('Attention Weight')
            ax_bot.legend(fontsize=7)

    fig.suptitle('Transformer Attention on CIR Waveforms')
    _savefig('13_attention_map.png')


def plot_annotated_cir(df, path1_idx, path1_amp, path2_idx, path2_amp,
                       cls_results, features_df, n=3):
    """
    Annotated CIR examples showing detected paths and predicted labels.

    Combines peak detection and classification results in a single
    visualisation -- demonstrating the end-to-end pipeline output.

    Parameters:
        df (pd.DataFrame): Preprocessed dataset with CIR columns and NLOS label.
        path1_idx, path1_amp: Path 1 detection results.
        path2_idx, path2_amp: Path 2 detection results.
        cls_results (dict): Classification results (best model is auto-selected).
        features_df (pd.DataFrame): Feature matrix (for potential feature display).
        n (int): Number of annotated examples to show (default 3).
    """
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    # Select the best classifier by accuracy for annotation
    best_cls = max(cls_results, key=lambda n: cls_results[n]['accuracy'])
    model = cls_results[best_cls]['model']

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    np.random.seed(42)  # Reproducible random sample selection
    sample_idxs = np.random.choice(len(df), n, replace=False)

    for ax, i in zip(axes, sample_idxs):
        cir = df.iloc[i][cir_cols].values.astype(float)
        true_label = 'NLOS' if df.iloc[i]['NLOS'] == 1 else 'LOS'

        ax.plot(cir, linewidth=0.5, color='gray')
        # Annotate Path 1 (blue downward triangle)
        ax.plot(path1_idx[i], path1_amp[i], 'bv', markersize=10, label='Path 1')
        # Annotate Path 2 if detected (red upward triangle)
        if path2_amp[i] > 0:
            ax.plot(path2_idx[i], path2_amp[i], 'r^', markersize=10, label='Path 2')

        # Zoom to region around detected paths
        ax.set_xlim(max(0, path1_idx[i] - 50), min(len(cir), path1_idx[i] + 100))
        ax.set_title(f'Sample {i} (True: {true_label})')
        ax.legend(fontsize=7)

    fig.suptitle(f'Annotated CIR Examples ({best_cls})')
    _savefig('12_annotated_cir.png')
