"""All visualization functions organized by 3D stages."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


PLOT_DIR = 'plots/'


def _savefig(name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Data Preparation Plots ──────────────────────────────────────────

def plot_class_distribution(original_nlos, expanded_labels):
    """Bar chart of class distribution before/after two-path expansion."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Original
    vals, counts = np.unique(original_nlos, return_counts=True)
    axes[0].bar(['LOS', 'NLOS'], counts, color=['steelblue', 'coral'])
    axes[0].set_title('Original Dataset')
    axes[0].set_ylabel('Count')
    for j, c in enumerate(counts):
        axes[0].text(j, c + 200, str(c), ha='center')

    # Expanded
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
    """Histograms/KDE of key scalar features by LOS/NLOS."""
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
    """Correlation heatmap of scalar features."""
    scalar = ['RANGE', 'FP_IDX', 'FP_AMP1', 'FP_AMP2', 'FP_AMP3',
              'STDEV_NOISE', 'CIR_PWR', 'MAX_NOISE', 'RXPACC',
              'FRAME_LEN', 'PREAM_LEN', 'NLOS']
    corr = df[scalar].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Scalar Feature Correlation')
    _savefig('03_correlation_heatmap.png')


def plot_cir_examples(df, path1_idx, path1_amp, path2_idx, path2_amp, n=4):
    """Example CIR plots for LOS vs NLOS with detected peaks."""
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))

    for row, (label, title) in enumerate([(0, 'LOS'), (1, 'NLOS')]):
        idxs = np.where(df['NLOS'].values == label)[0][:n]
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            cir = df.iloc[i][cir_cols].values.astype(float)
            ax.plot(cir, linewidth=0.5, color='gray')
            ax.axvline(path1_idx[i], color='blue', linestyle='--', alpha=0.7, label='Path 1')
            ax.plot(path1_idx[i], path1_amp[i], 'bv', markersize=8)
            if path2_amp[i] > 0:
                ax.axvline(path2_idx[i], color='red', linestyle='--', alpha=0.7, label='Path 2')
                ax.plot(path2_idx[i], path2_amp[i], 'r^', markersize=8)
            ax.set_title(f'{title} #{col + 1}')
            ax.set_xlim(max(0, path1_idx[i] - 50), min(len(cir), path1_idx[i] + 100))
            if col == 0:
                ax.set_ylabel('Amplitude')
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle('CIR Examples with Detected Peaks')
    _savefig('04_cir_examples.png')


def plot_fp_idx_distribution(df):
    """FP_IDX distribution by class."""
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


# ── Data Mining Plots ────────────────────────────────────────────────

def plot_feature_importance(importances, feature_names, top_n=20):
    """Feature importance bar chart from Random Forest."""
    top_n = min(top_n, len(importances))
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[idx][::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)')
    _savefig('06_feature_importance.png')


def plot_confusion_matrices(cls_results, y_test):
    """Confusion matrix heatmaps for all classifiers."""
    n_models = len(cls_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, cls_results.items()):
        # Use stored confusion matrix (supports models with different test sets)
        cm = res['confusion_matrix']
        ConfusionMatrixDisplay(cm, display_labels=['LOS', 'NLOS']).plot(
            cmap='Blues', ax=ax,
        )
        ax.set_title(name)
    fig.suptitle('Confusion Matrices')
    _savefig('07_confusion_matrices.png')


def plot_roc_curves(cls_results):
    """ROC curves for all classifiers overlaid."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in cls_results.items():
        ax.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    _savefig('08_roc_curves.png')


def plot_model_comparison(cls_results):
    """Bar chart comparing accuracy and F1 across models."""
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


# ── Results Plots ────────────────────────────────────────────────────

def plot_predicted_vs_actual(reg_results_p1, y_test_p1, reg_results_p2, y_test_p2):
    """Predicted vs actual scatter for best regressor on each path."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, y_test, title in [
        (axes[0], reg_results_p1, y_test_p1, 'Path 1 Range'),
        (axes[1], reg_results_p2, y_test_p2, 'Path 2 Range'),
    ]:
        # Pick best model by RMSE
        best_name = min(results, key=lambda n: results[n]['rmse'])
        y_pred = results[best_name]['y_pred']
        ax.scatter(y_test, y_pred, alpha=0.1, s=5, color='steelblue')
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel('Actual (m)')
        ax.set_ylabel('Predicted (m)')
        ax.set_title(f'{title} — {best_name}\n'
                      f'RMSE={results[best_name]["rmse"]:.3f}m')
    fig.suptitle('Predicted vs Actual Range')
    _savefig('10_predicted_vs_actual.png')


def plot_residuals(reg_results_p1, y_test_p1, reg_results_p2, y_test_p2):
    """Residual distribution histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, y_test, title in [
        (axes[0], reg_results_p1, y_test_p1, 'Path 1'),
        (axes[1], reg_results_p2, y_test_p2, 'Path 2'),
    ]:
        best_name = min(results, key=lambda n: results[n]['rmse'])
        residuals = y_test - results[best_name]['y_pred']
        ax.hist(residuals, bins=80, color='steelblue', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel('Residual (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'{title} Residuals — {best_name}')
    fig.suptitle('Residual Distributions')
    _savefig('11_residuals.png')


def plot_attention_map(dl_model, df, train_idx, test_idx, n=4):
    """Visualize transformer attention weights on sample CIR waveforms."""
    import torch
    from src.preprocessing import SCALAR_FEATURES

    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    device = next(dl_model.parameters()).device
    dl_model.eval()

    # Pick sample LOS and NLOS from test set
    test_labels = df['NLOS'].values[test_idx]
    los_idxs = test_idx[test_labels == 0][:n // 2]
    nlos_idxs = test_idx[test_labels == 1][:n // 2]
    sample_idxs = np.concatenate([los_idxs, nlos_idxs])

    fig, axes = plt.subplots(2, len(sample_idxs), figsize=(5 * len(sample_idxs), 8))

    for col, idx in enumerate(sample_idxs):
        cir_raw = df.iloc[idx][cir_cols].values.astype(float)
        cir_tensor = torch.tensor(cir_raw, dtype=torch.float32).unsqueeze(0).to(device)
        scalar_tensor = torch.tensor(
            df.iloc[idx][SCALAR_FEATURES].values.astype(float),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = dl_model(cir_tensor, scalar_tensor)
            pred = 'NLOS' if logit.item() > 0 else 'LOS'

        true_label = 'NLOS' if df.iloc[idx]['NLOS'] == 1 else 'LOS'

        # Get attention weights: [1, seq_len, seq_len] -> average over queries
        attn = dl_model.last_attention_weights
        if attn is not None:
            attn_avg = attn[0].mean(dim=0).cpu().numpy()  # [seq_len]
            # Upsample attention to CIR length
            attn_upsampled = np.interp(
                np.linspace(0, 1, len(cir_raw)),
                np.linspace(0, 1, len(attn_avg)),
                attn_avg,
            )
        else:
            attn_upsampled = np.zeros(len(cir_raw))

        # Top: CIR waveform with attention overlay
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

        # Bottom: attention heatmap
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
    """Annotated CIR examples with predicted labels and distances."""
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    best_cls = max(cls_results, key=lambda n: cls_results[n]['accuracy'])
    model = cls_results[best_cls]['model']

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    np.random.seed(42)
    sample_idxs = np.random.choice(len(df), n, replace=False)

    for ax, i in zip(axes, sample_idxs):
        cir = df.iloc[i][cir_cols].values.astype(float)
        true_label = 'NLOS' if df.iloc[i]['NLOS'] == 1 else 'LOS'

        ax.plot(cir, linewidth=0.5, color='gray')
        ax.plot(path1_idx[i], path1_amp[i], 'bv', markersize=10, label='Path 1')
        if path2_amp[i] > 0:
            ax.plot(path2_idx[i], path2_amp[i], 'r^', markersize=10, label='Path 2')

        ax.set_xlim(max(0, path1_idx[i] - 50), min(len(cir), path1_idx[i] + 100))
        ax.set_title(f'Sample {i} (True: {true_label})')
        ax.legend(fontsize=7)

    fig.suptitle(f'Annotated CIR Examples ({best_cls})')
    _savefig('12_annotated_cir.png')
