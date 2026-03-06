"""Main pipeline: UWB LOS/NLOS Classification & Distance Estimation."""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_dataset
from src.preprocessing import preprocess, scale_and_split, SCALAR_FEATURES
from src.peak_detection import extract_two_paths
from src.feature_engineering import build_features
from src.classification import train_classifiers
from src.regression import train_regressors
from src.dl_training import train_dl_classifier
from src import visualization as viz


def main():
    # ── Step 1: Load Data ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    df_raw = load_dataset()

    # ── Step 2: Preprocessing ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing")
    print("=" * 60)
    df = preprocess(df_raw)
    df_scaled, train_idx, test_idx, scaler = scale_and_split(df)

    # ── Step 3: Peak Detection ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Two Dominant Path Extraction")
    print("=" * 60)
    # Use unscaled data for peak detection (need real amplitudes)
    path1_idx, path1_amp, path2_idx, path2_amp = extract_two_paths(df)

    # ── Step 4: Feature Engineering ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Feature Engineering")
    print("=" * 60)
    features_df, labels_cls, labels_range, path_ids = build_features(
        df, path1_idx, path1_amp, path2_idx, path2_amp,
    )

    # Map original train/test indices to expanded indices
    n_orig = len(df)
    # path1 rows are 0..n_orig-1, path2 rows are n_orig..2*n_orig-1
    exp_train_idx = np.concatenate([train_idx, train_idx + n_orig])
    exp_test_idx = np.concatenate([test_idx, test_idx + n_orig])

    X_train_cls = features_df.iloc[exp_train_idx].values
    y_train_cls = labels_cls[exp_train_idx]
    X_test_cls = features_df.iloc[exp_test_idx].values
    y_test_cls = labels_cls[exp_test_idx]

    feature_names = list(features_df.columns)

    # ── Step 5: Classification ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: LOS/NLOS Classification")
    print("=" * 60)
    cls_results = train_classifiers(X_train_cls, y_train_cls, X_test_cls, y_test_cls)

    # ── Step 5b: Deep Learning on Raw CIR ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5b: CNN+Transformer on Raw CIR")
    print("=" * 60)

    # Prepare CIR and scalar tensors from original (non-expanded) data
    cir_cols = [c for c in df.columns if c.startswith('CIR') and c != 'CIR_PWR']
    X_cir_train = df_scaled[cir_cols].values[train_idx]
    X_cir_test = df_scaled[cir_cols].values[test_idx]
    X_scalar_train = df_scaled[SCALAR_FEATURES].values[train_idx]
    X_scalar_test = df_scaled[SCALAR_FEATURES].values[test_idx]
    y_dl_train = df_scaled['NLOS'].values[train_idx]
    y_dl_test = df_scaled['NLOS'].values[test_idx]

    dl_result = train_dl_classifier(
        X_cir_train, X_scalar_train, y_dl_train,
        X_cir_test, X_scalar_test, y_dl_test,
    )
    cls_results['CNN+Transformer'] = dl_result

    # ── Step 6: Distance Estimation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Distance Estimation")
    print("=" * 60)

    # Path 1 regression
    p1_mask_train = exp_train_idx[exp_train_idx < n_orig]
    p1_mask_test = exp_test_idx[exp_test_idx < n_orig]
    X_train_p1 = features_df.iloc[p1_mask_train].values
    y_train_p1 = labels_range[p1_mask_train]
    X_test_p1 = features_df.iloc[p1_mask_test].values
    y_test_p1 = labels_range[p1_mask_test]

    print("\n>> Path 1 Distance Estimation")
    reg_results_p1 = train_regressors(X_train_p1, y_train_p1, X_test_p1, y_test_p1, "Path 1")

    # Path 2 regression (include RANGE as additional feature)
    p2_mask_train = exp_train_idx[exp_train_idx >= n_orig] - n_orig
    p2_mask_test = exp_test_idx[exp_test_idx >= n_orig] - n_orig

    # Add original RANGE as feature for path 2
    X_train_p2_base = features_df.iloc[p2_mask_train + n_orig].values
    X_test_p2_base = features_df.iloc[p2_mask_test + n_orig].values
    range_train = df['RANGE'].values[p2_mask_train].reshape(-1, 1)
    range_test = df['RANGE'].values[p2_mask_test].reshape(-1, 1)
    X_train_p2 = np.hstack([X_train_p2_base, range_train])
    X_test_p2 = np.hstack([X_test_p2_base, range_test])
    y_train_p2 = labels_range[p2_mask_train + n_orig]
    y_test_p2 = labels_range[p2_mask_test + n_orig]

    # Filter out samples where path2 was not found (range would be unreliable)
    valid_train = path2_amp[p2_mask_train] > 0
    valid_test = path2_amp[p2_mask_test] > 0
    print(f"\n  Path 2 valid samples: train={valid_train.sum()}, test={valid_test.sum()}")

    print("\n>> Path 2 Distance Estimation")
    reg_results_p2 = train_regressors(
        X_train_p2[valid_train], y_train_p2[valid_train],
        X_test_p2[valid_test], y_test_p2[valid_test], "Path 2",
    )

    # ── Step 7: Visualization ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Generating Visualizations")
    print("=" * 60)

    # Data Preparation plots
    print("\n[Data Preparation]")
    viz.plot_class_distribution(df_raw['NLOS'].values, labels_cls)
    viz.plot_feature_distributions(df_raw)
    viz.plot_correlation_heatmap(df_raw)
    viz.plot_cir_examples(df, path1_idx, path1_amp, path2_idx, path2_amp)
    viz.plot_fp_idx_distribution(df_raw)

    # Data Mining plots
    print("\n[Data Mining]")
    if 'feature_importances' in cls_results.get('Random Forest', {}):
        viz.plot_feature_importance(
            cls_results['Random Forest']['feature_importances'],
            feature_names,
        )
    viz.plot_confusion_matrices(cls_results, y_test_cls)
    viz.plot_roc_curves(cls_results)
    viz.plot_model_comparison(cls_results)

    if 'CNN+Transformer' in cls_results:
        viz.plot_attention_map(
            cls_results['CNN+Transformer']['model'],
            df_scaled, train_idx, test_idx,
        )

    # Results plots
    print("\n[Results]")
    viz.plot_predicted_vs_actual(reg_results_p1, y_test_p1,
                                 reg_results_p2, y_test_p2[valid_test])
    viz.plot_residuals(reg_results_p1, y_test_p1,
                       reg_results_p2, y_test_p2[valid_test])
    viz.plot_annotated_cir(df, path1_idx, path1_amp, path2_idx, path2_amp,
                           cls_results, features_df)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nClassification Results:")
    for name, res in cls_results.items():
        print(f"  {name}: Accuracy={res['accuracy']:.4f}, AUC={res['auc']:.4f}")

    print("\nPath 1 Distance Estimation:")
    for name, res in reg_results_p1.items():
        print(f"  {name}: RMSE={res['rmse']:.4f}m, R²={res['r2']:.4f}")

    print("\nPath 2 Distance Estimation:")
    for name, res in reg_results_p2.items():
        print(f"  {name}: RMSE={res['rmse']:.4f}m, R²={res['r2']:.4f}")

    print(f"\nAll plots saved to {viz.PLOT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
