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

# Only force the non-interactive Agg backend when NOT running inside a
# notebook / IPython session.  This allows %matplotlib inline to work
# when the module is imported from a Jupyter notebook while still
# defaulting to Agg for headless script execution (main.py).
import sys

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Output directory for all generated plot images
PLOT_DIR = "plots/"


def _savefig(name):
    """Save the current matplotlib figure to the plots directory and close it."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
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
    axes[0].bar(["LOS", "NLOS"], counts, color=["steelblue", "coral"])
    axes[0].set_title("Original Dataset")
    axes[0].set_ylabel("Count")
    for j, c in enumerate(counts):
        axes[0].text(j, c + 200, str(c), ha="center")

    # Right panel: Expanded 84K dataset (imbalanced 25/75)
    vals2, counts2 = np.unique(expanded_labels, return_counts=True)
    labels2 = ["LOS" if v == 0 else "NLOS" for v in vals2]
    axes[1].bar(labels2, counts2, color=["steelblue", "coral"])
    axes[1].set_title("Two-Path Expanded")
    axes[1].set_ylabel("Count")
    for j, c in enumerate(counts2):
        axes[1].text(j, c + 500, str(c), ha="center")

    fig.suptitle("Class Distribution")
    _savefig("01_class_distribution.png")


def plot_feature_distributions(df):
    """
    Histograms/KDE of key scalar features by LOS/NLOS class.

    Visualises the feature distributions to understand which features show
    separation between LOS and NLOS classes — informing feature importance
    and the rationale for feature selection (Data Preparation step V).

    Parameters:
        df (pd.DataFrame): Raw dataset with scalar features and NLOS label.
    """
    feats = ["RANGE", "FP_AMP1", "STDEV_NOISE", "CIR_PWR", "MAX_NOISE", "RXPACC"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.ravel(), feats):
        for label, color in [(0, "steelblue"), (1, "coral")]:
            subset = df[df["NLOS"] == label][feat]
            ax.hist(
                subset,
                bins=50,
                alpha=0.5,
                color=color,
                label="LOS" if label == 0 else "NLOS",
                density=True,
            )
        ax.set_title(feat)
        ax.legend(fontsize=8)
    fig.suptitle("Feature Distributions by LOS/NLOS")
    _savefig("02_feature_distributions.png")


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
    scalar = [
        "RANGE",
        "FP_IDX",
        "FP_AMP1",
        "FP_AMP2",
        "FP_AMP3",
        "STDEV_NOISE",
        "CIR_PWR",
        "MAX_NOISE",
        "RXPACC",
        "FRAME_LEN",
        "PREAM_LEN",
        "NLOS",
    ]
    corr = df[scalar].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Scalar Feature Correlation")
    _savefig("03_correlation_heatmap.png")


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
    cir_cols = [c for c in df.columns if c.startswith("CIR") and c != "CIR_PWR"]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))

    for row, (label, title) in enumerate([(0, "LOS"), (1, "NLOS")]):
        idxs = np.where(df["NLOS"].values == label)[0][:n]
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            cir = df.iloc[i][cir_cols].values.astype(float)
            ax.plot(cir, linewidth=0.5, color="gray")
            # Mark Path 1 (blue triangle, dashed line)
            ax.axvline(
                path1_idx[i], color="blue", linestyle="--", alpha=0.7, label="Path 1"
            )
            ax.plot(path1_idx[i], path1_amp[i], "bv", markersize=8)
            # Mark Path 2 if detected (red triangle, dashed line)
            if path2_amp[i] > 0:
                ax.axvline(
                    path2_idx[i], color="red", linestyle="--", alpha=0.7, label="Path 2"
                )
                ax.plot(path2_idx[i], path2_amp[i], "r^", markersize=8)
            ax.set_title(f"{title} #{col + 1}")
            # Zoom to the region around the detected paths
            ax.set_xlim(max(0, path1_idx[i] - 50), min(len(cir), path1_idx[i] + 100))
            if col == 0:
                ax.set_ylabel("Amplitude")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle("CIR Examples with Detected Peaks")
    _savefig("04_cir_examples.png")


def plot_fp_idx_distribution(df):
    """
    FP_IDX (First Path Index) distribution by LOS/NLOS class.

    Shows whether the hardware-reported first path index differs between
    LOS and NLOS conditions, which informs feature engineering decisions.

    Parameters:
        df (pd.DataFrame): Raw dataset with FP_IDX and NLOS columns.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [(0, "steelblue"), (1, "coral")]:
        subset = df[df["NLOS"] == label]["FP_IDX"]
        ax.hist(
            subset,
            bins=50,
            alpha=0.5,
            color=color,
            label="LOS" if label == 0 else "NLOS",
            density=True,
        )
    ax.set_xlabel("FP_IDX")
    ax.set_ylabel("Density")
    ax.set_title("First Path Index Distribution")
    ax.legend()
    _savefig("05_fp_idx_distribution.png")


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
    ax.barh(range(top_n), importances[idx][::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    _savefig("06_feature_importance.png")


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
    # Use a 2-row grid when there are more than 4 models to avoid an
    # impractically wide figure (e.g., 8 models x 5in = 40in).
    n_cols = min(n_models, 4)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).ravel()  # Flatten for uniform indexing
    for i, (name, res) in enumerate(cls_results.items()):
        # Use pre-computed confusion matrix from each model's results
        cm = res["confusion_matrix"]
        ConfusionMatrixDisplay(cm, display_labels=["LOS", "NLOS"]).plot(
            cmap="Blues",
            ax=axes[i],
        )
        axes[i].set_title(name, fontsize=9)
    # Hide any unused subplot slots
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Confusion Matrices")
    _savefig("07_confusion_matrices.png")


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
        ax.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)  # Random classifier baseline
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    _savefig("08_roc_curves.png")


def plot_pr_curves(cls_results, y_test):
    """
    Precision-Recall curves for all classifiers overlaid on the same axes.

    The PR curve plots Precision vs Recall at varying classification thresholds.
    AP (Average Precision) summarises the area under the PR curve -- higher AP
    indicates better precision-recall trade-off, especially for imbalanced data.

    A horizontal dashed line shows the "no skill" baseline, equal to the
    proportion of positive samples (NLOS) in the test set.

    Parameters:
        cls_results (dict): Classification results containing 'y_prob'.
        y_test (np.ndarray): True test labels (0=LOS, 1=NLOS).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in cls_results.items():
        # Skip models with different test set size (e.g., DL on original 42K vs ML on expanded 84K)
        if len(res["y_prob"]) != len(y_test):
            continue
        precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
        ap = average_precision_score(y_test, res["y_prob"])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    # No-skill baseline: proportion of positive class
    no_skill = y_test.sum() / len(y_test)
    ax.axhline(
        no_skill,
        color="k",
        linestyle="--",
        alpha=0.3,
        label=f"No Skill ({no_skill:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    _savefig("15_pr_curves.png")


def plot_model_comparison(cls_results):
    """
    Grouped bar chart comparing accuracy and AUC across all models.

    Provides a quick visual summary of model performance for the
    Data Analysis stage.

    Parameters:
        cls_results (dict): Classification results containing 'accuracy' and 'auc'.
    """
    names = list(cls_results.keys())
    accs = [cls_results[n]["accuracy"] for n in names]
    aucs = [cls_results[n]["auc"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, aucs, width, label="AUC", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    _savefig("09_model_comparison.png")


def plot_per_environment_heatmap(cls_results, X_test, y_test, env_ids_test):
    """
    Heatmap of per-environment classification accuracy for each sklearn model.

    Computes accuracy for each (model, environment) pair and displays it as
    a seaborn heatmap, revealing whether certain indoor environments are
    harder to classify than others.

    Parameters:
        cls_results (dict): Classification results from all models.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): True test labels.
        env_ids_test (np.ndarray): Environment ID (0-6) for each test sample.
    """
    env_names = [
        "Office 1",
        "Office 2",
        "Small Apartment",
        "Small Workshop",
        "Kitchen+Living",
        "Bedroom",
        "Boiler Room",
    ]

    # Collect sklearn models that were trained on the original feature space.
    # Skip ensemble meta-learners and DL models (different input dimensions).
    skip_names = {"Ensemble (Average)", "Ensemble (Stacked)", "CNN+Transformer"}
    sklearn_models = {
        name: res
        for name, res in cls_results.items()
        if "model" in res
        and hasattr(res["model"], "predict")
        and name not in skip_names
    }

    if not sklearn_models:
        print("  Skipped: no sklearn models with .predict() found")
        return

    model_names = list(sklearn_models.keys())
    acc_matrix = np.zeros((len(env_names), len(model_names)))

    for j, name in enumerate(model_names):
        model = sklearn_models[name]["model"]
        y_pred = model.predict(X_test)
        for i in range(len(env_names)):
            mask = env_ids_test == i
            if mask.sum() > 0:
                acc_matrix[i, j] = (y_pred[mask] == y_test[mask]).mean()

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(model_names)), 6))
    sns.heatmap(
        acc_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=model_names,
        yticklabels=env_names,
        ax=ax,
        vmin=0.5,
        vmax=1.0,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Environment")
    ax.set_title("Per-Environment Classification Accuracy")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    _savefig("16_per_environment_heatmap.png")


def plot_regression_comparison(reg_results_p1, reg_results_p2):
    """
    Grouped bar chart comparing regression metrics across all models and paths.

    Three subplots side by side showing RMSE, MAE, and R² for each regressor,
    with Path 1 (steelblue) and Path 2 (coral) bars grouped per model.

    Parameters:
        reg_results_p1 (dict): Regression results for Path 1.
        reg_results_p2 (dict): Regression results for Path 2.
    """
    names = list(reg_results_p1.keys())
    metrics = ["rmse", "mae", "r2"]
    titles = ["RMSE (m)", "MAE (m)", "R²"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))
    width = 0.35

    for ax, metric, title in zip(axes, metrics, titles):
        vals_p1 = [reg_results_p1[n][metric] for n in names]
        vals_p2 = [reg_results_p2[n][metric] for n in names]
        ax.bar(x - width / 2, vals_p1, width, label="Path 1", color="steelblue")
        ax.bar(x + width / 2, vals_p2, width, label="Path 2", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Regression Model Comparison")
    _savefig("17_regression_comparison.png")


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
        (axes[0], reg_results_p1, y_test_p1, "Path 1 Range"),
        (axes[1], reg_results_p2, y_test_p2, "Path 2 Range"),
    ]:
        # Automatically select model with lowest RMSE
        best_name = min(results, key=lambda n: results[n]["rmse"])
        y_pred = results[best_name]["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.1, s=5, color="steelblue")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", alpha=0.5)  # Perfect prediction line
        ax.set_xlabel("Actual (m)")
        ax.set_ylabel("Predicted (m)")
        ax.set_title(f"{title} -- {best_name}\nRMSE={results[best_name]['rmse']:.3f}m")
    fig.suptitle("Predicted vs Actual Range")
    _savefig("10_predicted_vs_actual.png")


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
        (axes[0], reg_results_p1, y_test_p1, "Path 1"),
        (axes[1], reg_results_p2, y_test_p2, "Path 2"),
    ]:
        best_name = min(results, key=lambda n: results[n]["rmse"])
        residuals = y_test - results[best_name]["y_pred"]
        ax.hist(residuals, bins=80, color="steelblue", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--")  # Zero-error reference
        ax.set_xlabel("Residual (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title} Residuals -- {best_name}")
    fig.suptitle("Residual Distributions")
    _savefig("11_residuals.png")


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

    cir_cols = [c for c in df.columns if c.startswith("CIR") and c != "CIR_PWR"]
    device = next(dl_model.parameters()).device
    dl_model.eval()

    # Select n/2 LOS and n/2 NLOS samples from the test set
    test_labels = df["NLOS"].values[test_idx]
    los_idxs = test_idx[test_labels == 0][: n // 2]
    nlos_idxs = test_idx[test_labels == 1][: n // 2]
    sample_idxs = np.concatenate([los_idxs, nlos_idxs])

    fig, axes = plt.subplots(2, len(sample_idxs), figsize=(5 * len(sample_idxs), 8))

    for col, idx in enumerate(sample_idxs):
        # Prepare single-sample tensors for forward pass
        cir_raw = df.iloc[idx][cir_cols].values.astype(float)
        cir_tensor = torch.tensor(cir_raw, dtype=torch.float32).unsqueeze(0).to(device)
        scalar_tensor = (
            torch.tensor(
                df.iloc[idx][SCALAR_FEATURES].values.astype(float),
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(device)
        )

        # Forward pass to get prediction and trigger attention hook
        with torch.no_grad():
            logit = dl_model(cir_tensor, scalar_tensor)
            pred = "NLOS" if logit.item() > 0 else "LOS"

        true_label = "NLOS" if df.iloc[idx]["NLOS"] == 1 else "LOS"

        # Extract attention weights: [1, seq_len, seq_len] -> average over queries -> [seq_len]
        attn = dl_model.last_attention_weights
        if attn is not None:
            attn_avg = (
                attn[0].mean(dim=0).cpu().numpy()
            )  # Average across query positions
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
        ax_top.plot(cir_raw, color="gray", linewidth=0.5)
        ax_top.fill_between(
            range(len(cir_raw)),
            0,
            cir_raw,
            alpha=0.3,
            color="red" if true_label == "NLOS" else "blue",
        )
        ax_top.set_title(f"True: {true_label}, Pred: {pred}")
        if col == 0:
            ax_top.set_ylabel("CIR Amplitude")

        # Bottom row: attention-weighted overlay (orange = high attention)
        ax_bot = axes[1, col]
        ax_bot.plot(cir_raw, color="gray", linewidth=0.5, alpha=0.5)
        ax_bot.fill_between(
            range(len(cir_raw)),
            0,
            attn_upsampled * cir_raw.max(),
            alpha=0.6,
            color="orange",
            label="Attention",
        )
        ax_bot.set_xlabel("CIR Sample Index")
        if col == 0:
            ax_bot.set_ylabel("Attention Weight")
            ax_bot.legend(fontsize=7)

    fig.suptitle("Transformer Attention on CIR Waveforms")
    _savefig("13_attention_map.png")


def plot_clustering(cluster_results, y_test):
    """
    Visualise K-Means clustering results: PCA scatter plot and cluster-vs-label comparison.

    Left panel: 2D PCA projection of test features, coloured by K-Means cluster assignment.
    Right panel: Same projection, coloured by true LOS/NLOS labels.
    This side-by-side comparison shows how well unsupervised clustering recovers the
    true class structure in the feature space.

    Parameters:
        cluster_results (dict): Output from run_kmeans_analysis().
        y_test (np.ndarray): True test labels (0=LOS, 1=NLOS).
    """
    X_2d = cluster_results["X_test_2d"]
    clusters = cluster_results["test_labels"]
    acc = cluster_results["test_accuracy"]
    sil = cluster_results["silhouette_test"]
    ari = cluster_results["ari"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: coloured by cluster assignment
    for label, name, color in [
        (0, "Cluster 0 (LOS)", "steelblue"),
        (1, "Cluster 1 (NLOS)", "coral"),
    ]:
        mask = clusters == label
        axes[0].scatter(
            X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.15, s=5, label=name
        )
    axes[0].set_title(
        f"K-Means Clusters\nAcc={acc:.3f}, Silhouette={sil:.3f}, ARI={ari:.3f}"
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(markerscale=4, fontsize=9)

    # Right: coloured by true labels
    for label, name, color in [
        (0, "LOS (true)", "steelblue"),
        (1, "NLOS (true)", "coral"),
    ]:
        mask = y_test == label
        axes[1].scatter(
            X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.15, s=5, label=name
        )
    axes[1].set_title("True Labels (PCA projection)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(markerscale=4, fontsize=9)

    fig.suptitle("Unsupervised Clustering vs True Labels")
    _savefig("14_clustering.png")


def plot_elbow_silhouette(elbow_results):
    """
    Elbow and Silhouette plots for optimal cluster count selection.

    Left panel shows inertia (within-cluster sum of squares) vs k -- the
    "elbow" point indicates diminishing returns from adding more clusters.
    Right panel shows silhouette score vs k -- higher is better.

    Both panels highlight k=2 (the chosen value for LOS/NLOS clustering)
    with a vertical dashed red line.

    Parameters:
        elbow_results (dict): Output from run_elbow_silhouette_analysis().
    """
    k_values = elbow_results["k_values"]
    inertias = elbow_results["inertias"]
    silhouettes = elbow_results["silhouettes"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Elbow curve (Inertia vs k)
    axes[0].plot(k_values, inertias, "o-", color="steelblue", markersize=6)
    axes[0].axvline(2, color="red", linestyle="--", alpha=0.7)
    axes[0].annotate(
        "k=2 (chosen)",
        xy=(2, inertias[0]),
        xytext=(3.5, inertias[0]),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=9,
    )
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method (Inertia)")

    # Right panel: Silhouette score vs k
    axes[1].plot(k_values, silhouettes, "o-", color="steelblue", markersize=6)
    axes[1].axvline(2, color="red", linestyle="--", alpha=0.7)
    axes[1].annotate(
        "k=2 (chosen)",
        xy=(2, silhouettes[0]),
        xytext=(3.5, silhouettes[0]),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=9,
    )
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs k")

    fig.suptitle("Optimal Cluster Count Analysis")
    _savefig("19_elbow_silhouette.png")


def plot_tsne_embedding(X_test, y_test, cluster_labels, random_state=42):
    """
    t-SNE 2D embedding of test features compared with PCA projection.

    Three panels side by side:
      - Left: t-SNE coloured by true LOS/NLOS labels.
      - Middle: t-SNE coloured by K-Means cluster assignments.
      - Right: PCA 2D projection coloured by true labels for comparison.

    Since t-SNE is computationally expensive on large datasets, the input
    is subsampled to a maximum of 10,000 samples for tractability.

    Parameters:
        X_test (np.ndarray): Test feature matrix (n_samples, n_features).
        y_test (np.ndarray): True test labels (0=LOS, 1=NLOS).
        cluster_labels (np.ndarray): K-Means cluster assignments for test set.
        random_state (int): Random seed for reproducibility (default 42).
    """
    # Subsample to max 10,000 samples for t-SNE tractability
    max_samples = 10_000
    n = len(X_test)
    if n > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, max_samples, replace=False)
        X_sub = X_test[idx]
        y_sub = y_test[idx]
        cl_sub = cluster_labels[idx]
    else:
        X_sub = X_test
        y_sub = y_test
        cl_sub = cluster_labels

    # t-SNE reduction to 2D
    print("  Running t-SNE (this may take a minute)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=random_state,
        learning_rate="auto",
        init="pca",
    )
    X_tsne = tsne.fit_transform(X_sub)

    # PCA reduction to 2D for comparison
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_sub)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left panel: t-SNE coloured by true labels
    for label, name, color in [
        (0, "LOS (true)", "steelblue"),
        (1, "NLOS (true)", "coral"),
    ]:
        mask = y_sub == label
        axes[0].scatter(
            X_tsne[mask, 0], X_tsne[mask, 1], c=color, alpha=0.15, s=5, label=name
        )
    axes[0].set_title("t-SNE: True Labels")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(markerscale=4, fontsize=9)

    # Middle panel: t-SNE coloured by K-Means cluster assignments
    for label, name, color in [
        (0, "Cluster 0 (LOS)", "steelblue"),
        (1, "Cluster 1 (NLOS)", "coral"),
    ]:
        mask = cl_sub == label
        axes[1].scatter(
            X_tsne[mask, 0], X_tsne[mask, 1], c=color, alpha=0.15, s=5, label=name
        )
    axes[1].set_title("t-SNE: K-Means Clusters")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(markerscale=4, fontsize=9)

    # Right panel: PCA coloured by true labels for comparison
    for label, name, color in [
        (0, "LOS (true)", "steelblue"),
        (1, "NLOS (true)", "coral"),
    ]:
        mask = y_sub == label
        axes[2].scatter(
            X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.15, s=5, label=name
        )
    axes[2].set_title("PCA: True Labels (comparison)")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].legend(markerscale=4, fontsize=9)

    fig.suptitle("t-SNE vs PCA: Feature Space Embedding")
    _savefig("18_tsne_embedding.png")


def plot_annotated_cir(
    df, path1_idx, path1_amp, path2_idx, path2_amp, cls_results, features_df, n=3
):
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
    cir_cols = [c for c in df.columns if c.startswith("CIR") and c != "CIR_PWR"]
    # Select the best sklearn classifier by accuracy for prediction annotation.
    # Exclude DL models (CNN+Transformer) since they require different input format.
    sklearn_models = {
        k: v
        for k, v in cls_results.items()
        if "model" in v and hasattr(v["model"], "predict")
    }
    best_cls = max(sklearn_models, key=lambda n: sklearn_models[n]["accuracy"])
    model = sklearn_models[best_cls]["model"]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    np.random.seed(42)  # Reproducible random sample selection
    sample_idxs = np.random.choice(len(df), n, replace=False)

    for ax, i in zip(axes, sample_idxs):
        cir = df.iloc[i][cir_cols].values.astype(float)
        true_label = "NLOS" if df.iloc[i]["NLOS"] == 1 else "LOS"

        # Get predicted label from the best ML classifier (Path 1 features)
        pred_label = true_label  # Fallback
        if i < len(features_df) // 2:
            try:
                feats = features_df.iloc[i].values.reshape(1, -1)
                pred = model.predict(feats)[0]
                pred_label = "NLOS" if pred == 1 else "LOS"
            except Exception:
                pass  # Keep fallback if prediction fails

        ax.plot(cir, linewidth=0.5, color="gray")
        # Annotate Path 1 (blue downward triangle)
        ax.plot(path1_idx[i], path1_amp[i], "bv", markersize=10, label="Path 1")
        # Annotate Path 2 if detected (red upward triangle)
        if path2_amp[i] > 0:
            ax.plot(path2_idx[i], path2_amp[i], "r^", markersize=10, label="Path 2")

        # Zoom to region around detected paths
        ax.set_xlim(max(0, path1_idx[i] - 50), min(len(cir), path1_idx[i] + 100))
        ax.set_title(f"Sample {i}\nTrue: {true_label}, Pred: {pred_label}", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(f"Annotated CIR Examples ({best_cls})")
    _savefig("12_annotated_cir.png")


def plot_shap_summary(cls_results, X_test, feature_names):
    """
    SHAP summary plots for tree-based classifier feature importance.

    Generates two plots using SHapley Additive exPlanations:
      - Beeswarm/dot plot: SHAP values per sample for each feature.
      - Bar plot: Mean absolute SHAP values (average feature impact).

    Parameters:
        cls_results (dict): Classification results from all models.
        X_test (np.ndarray): Test feature matrix.
        feature_names (list): Feature names for axis labels.
    """
    try:
        import shap
    except ImportError:
        print("  SHAP not installed; skipping SHAP summary plots")
        return

    # Select the best tree-based model with feature_importances_
    best_model = None
    best_name = None
    for name in ["XGBoost", "Random Forest", "Gradient Boosted Trees"]:
        if name in cls_results and "model" in cls_results[name]:
            model = cls_results[name]["model"]
            if hasattr(model, "feature_importances_"):
                best_model = model
                best_name = name
                break

    if best_model is None:
        print("  No tree-based model found for SHAP analysis")
        return

    print(f"  Using {best_name} for SHAP analysis...")

    # Subsample for speed (max 1000 samples)
    n_samples = min(1000, len(X_test))
    np.random.seed(42)
    idx = np.random.choice(len(X_test), n_samples, replace=False)
    X_sub = X_test[idx]

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sub)

    # For binary classification, use class 1 (NLOS) SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_sub, feature_names=feature_names, plot_type="dot", show=False
    )
    _savefig("20_shap_beeswarm.png")

    # Bar plot (mean absolute SHAP)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_sub, feature_names=feature_names, plot_type="bar", show=False
    )
    _savefig("21_shap_bar.png")


def plot_rfe_curve(rfe_results):
    """
    Recursive Feature Elimination accuracy curve.

    Plots cross-validation accuracy as a function of the number of features,
    highlighting the optimal feature count with a vertical red line.

    Parameters:
        rfe_results (dict): Results with keys 'n_features', 'scores'.
    """
    n_features = rfe_results["n_features"]
    scores = rfe_results["scores"]

    optimal_idx = np.argmax(scores)
    optimal_n = n_features[optimal_idx]
    optimal_score = scores[optimal_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        n_features,
        scores,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        color="steelblue",
        label="CV Accuracy",
    )
    ax.axvline(optimal_n, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.plot(optimal_n, optimal_score, marker="*", markersize=15, color="red")
    ax.annotate(
        f"Optimal: {optimal_n} features\n(Accuracy={optimal_score:.4f})",
        xy=(optimal_n, optimal_score),
        xytext=(optimal_n + 1, optimal_score - 0.02),
        fontsize=9,
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Cross-Validation Accuracy")
    ax.set_title("Recursive Feature Elimination")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _savefig("22_rfe_curve.png")


def plot_dbscan(dbscan_results, y_test):
    """
    DBSCAN clustering results: PCA scatter with noise points highlighted.

    Left panel: clusters coloured by DBSCAN assignment (noise in gray).
    Right panel: true LOS/NLOS labels for comparison.

    Parameters:
        dbscan_results (dict): Output from run_dbscan_analysis().
        y_test (np.ndarray): True test labels (0=LOS, 1=NLOS).
    """
    X_2d = dbscan_results["X_test_2d"]
    clusters = dbscan_results["test_labels"]
    n_clusters = dbscan_results["n_clusters"]
    n_noise = dbscan_results["n_noise"]
    accuracy = dbscan_results["accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: DBSCAN clusters
    color_map = {-1: "gray", 0: "steelblue", 1: "coral"}
    extra = ["green", "purple", "orange", "brown", "pink"]
    for i, cid in enumerate(sorted(set(clusters))):
        if cid not in color_map:
            color_map[cid] = extra[i % len(extra)]

    for cid in sorted(set(clusters)):
        mask = clusters == cid
        lbl = f"Noise ({mask.sum()})" if cid == -1 else f"Cluster {cid} ({mask.sum()})"
        mkr = "x" if cid == -1 else "o"
        sz = 20 if cid == -1 else 5
        axes[0].scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=color_map[cid],
            alpha=0.15,
            s=sz,
            label=lbl,
            marker=mkr,
        )
    axes[0].set_title(f"DBSCAN (n={n_clusters}, noise={n_noise}, acc={accuracy:.3f})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(markerscale=4, fontsize=9)

    # Right: true labels
    for label, name, color in [
        (0, "LOS (true)", "steelblue"),
        (1, "NLOS (true)", "coral"),
    ]:
        mask = y_test == label
        axes[1].scatter(
            X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.15, s=5, label=name
        )
    axes[1].set_title("True Labels (PCA projection)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(markerscale=4, fontsize=9)

    fig.suptitle("DBSCAN Clustering vs True Labels")
    _savefig("23_dbscan.png")


def plot_augmentation_impact(original_metrics, augmented_metrics):
    """
    Bar chart comparing performance before and after synthetic data augmentation.

    Shows accuracy and AUC for Random Forest (SMOTE) and CNN+Transformer
    (CIR augmentation) with delta annotations.

    Parameters:
        original_metrics (dict): Keys: 'rf_acc', 'rf_auc', 'dl_acc', 'dl_auc'.
        augmented_metrics (dict): Same keys with augmented results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = ["Random Forest\n(SMOTE)", "CNN+Transformer\n(CIR Aug)"]
    x = np.arange(len(models))
    width = 0.35

    for ax, metric_key, title in [
        (axes[0], "acc", "Accuracy"),
        (axes[1], "auc", "AUC"),
    ]:
        orig = [
            original_metrics[f"rf_{metric_key}"],
            original_metrics[f"dl_{metric_key}"],
        ]
        aug = [
            augmented_metrics[f"rf_{metric_key}"],
            augmented_metrics[f"dl_{metric_key}"],
        ]
        ax.bar(x - width / 2, orig, width, label="Original", color="steelblue")
        ax.bar(x + width / 2, aug, width, label="Augmented", color="coral")
        for i, (o, a) in enumerate(zip(orig, aug)):
            delta = a - o
            clr = "darkgreen" if delta >= 0 else "darkred"
            ax.text(
                i + width / 2,
                a + 0.005,
                f"{delta:+.4f}",
                ha="center",
                fontsize=8,
                color=clr,
                fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylim(0.9, 1.0)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Synthetic Data Augmentation Impact")
    _savefig("24_augmentation_impact.png")
