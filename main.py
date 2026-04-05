"""
main.py — Main pipeline: UWB LOS/NLOS Classification & Distance Estimation.

This script orchestrates the complete 3D Data Analytics pipeline:
  Step 1: Data Loading — Load 42,000 UWB CIR measurements from 7 indoor environments.
  Step 2: Data Preparation — Clean, normalize CIR, scale features, split 80/20.
  Step 3: Peak Detection — Extract two dominant propagation paths from each CIR.
  Step 4: Feature Engineering — Build 23-feature vectors for each path (84K rows).
  Step 5: Classification — Train LR, RF, GBT, XGBoost classifiers on hand-crafted features.
  Step 5b: Deep Learning — Train CNN+Transformer on raw 1016-sample CIR waveforms.
  Step 5c: Synthetic Data — SMOTE augmentation for ML + CIR augmentation for DL,
           with comparison against the original (non-augmented) results.
  Step 5d: Unsupervised Clustering — K-Means baseline on hand-crafted features.
  Step 6: Distance Estimation — Train Ridge, RF, GBT regressors for range prediction.
  Step 7: Visualization — Generate 13+ plots covering all 3D analytics stages.

Usage: python main.py

Libraries: numpy, and all project modules under src/
"""

from src import visualization as viz
from src.clustering import run_kmeans_analysis, run_elbow_silhouette_analysis, run_dbscan_analysis
from src.config import (
    RANDOM_STATE, DL_SEED, SMOTE_TARGET_RATIO, RF_SMOTE_N_ESTIMATORS,
    CIR_AUG_FACTOR, CIR_NOISE_LEVEL, CIR_MAX_SHIFT, CIR_SCALE_RANGE,
)
from src.ensemble import build_ensemble
from src.synthetic_data import apply_smote, generate_augmented_cir, evaluate_synthetic_impact
from src.dl_training import train_dl_classifier
from src.regression import train_regressors
from src.classification import train_classifiers
from src.feature_engineering import build_features
from src.peak_detection import extract_two_paths
from src.preprocessing import preprocess, scale_and_split, SCALAR_FEATURES
from src.data_loader import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score, roc_curve, auc as sk_auc, confusion_matrix,
    accuracy_score as acc_fn, roc_curve as roc_fn, auc as auc_fn,
)
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import project modules for each pipeline stage


def main():
    # ── Step 1: Load Data ────────────────────────────────────────────
    # Load all 7 environment CSVs into a single DataFrame (42K x 1031)
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    df_raw = load_dataset()
    env_ids = df_raw['ENV_ID'].values  # Track environment source (0-6) before preprocessing drops it

    # ── Step 2: Preprocessing ────────────────────────────────────────
    # Drop constant columns, normalize CIR by RXPACC, StandardScale
    # scalar features, stratified 80/20 train/test split
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing")
    print("=" * 60)
    df = preprocess(df_raw)
    df_scaled, train_idx, test_idx, scaler = scale_and_split(df)

    # ── Step 3: Peak Detection ───────────────────────────────────────
    # For each CIR sample, find the two dominant propagation paths:
    # Path 1 near FP_IDX, Path 2 as the next strongest peak
    print("\n" + "=" * 60)
    print("STEP 3: Two Dominant Path Extraction")
    print("=" * 60)
    # Use preprocessed (CIR normalized by RXPACC) but unscaled data for peak detection
    path1_idx, path1_amp, path2_idx, path2_amp = extract_two_paths(df)

    # ── Step 4: Feature Engineering ──────────────────────────────────
    # Extract 18 features per path and expand to 84K two-path dataset
    # with appropriate NLOS labels and distance labels
    print("\n" + "=" * 60)
    print("STEP 4: Feature Engineering")
    print("=" * 60)
    features_df, labels_cls, labels_range, path_ids = build_features(
        df, path1_idx, path1_amp, path2_idx, path2_amp,
    )

    # Map original train/test indices to expanded (two-path) indices.
    # Path 1 rows are at positions 0..N-1, Path 2 rows are at N..2N-1
    n_orig = len(df)
    exp_train_idx = np.concatenate([train_idx, train_idx + n_orig])
    exp_test_idx = np.concatenate([test_idx, test_idx + n_orig])

    # Expand environment IDs to match two-path dataset
    exp_env_ids = np.concatenate([env_ids, env_ids])
    env_ids_test = exp_env_ids[exp_test_idx]

    # Prepare ML classification data (expanded 84K dataset)
    X_train_cls = features_df.iloc[exp_train_idx].values
    y_train_cls = labels_cls[exp_train_idx]
    X_test_cls = features_df.iloc[exp_test_idx].values
    y_test_cls = labels_cls[exp_test_idx]

    feature_names = list(features_df.columns)

    # ── Step 5: Classification ───────────────────────────────────────
    # Train Logistic Regression, Random Forest (GridSearchCV),
    # and Gradient Boosted Trees (GridSearchCV) on 18 hand-crafted features
    print("\n" + "=" * 60)
    print("STEP 5: LOS/NLOS Classification")
    print("=" * 60)
    cls_results = train_classifiers(
        X_train_cls, y_train_cls, X_test_cls, y_test_cls)

    # ── Step 5a: Recursive Feature Elimination ──────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5a: Recursive Feature Elimination (RFE)")
    print("=" * 60)
    best_rf = cls_results['Random Forest']['model']
    rfecv = RFECV(
        estimator=best_rf, step=1, cv=3, scoring='accuracy',
        min_features_to_select=1, n_jobs=-1,
    )
    rfecv.fit(X_train_cls, y_train_cls)
    rfe_results = {
        'n_features': np.arange(1, len(rfecv.ranking_) + 1),
        'scores': rfecv.cv_results_['mean_test_score'],
    }
    print(f"  Optimal number of features: {rfecv.n_features_}")
    ranked = sorted(zip(feature_names, rfecv.ranking_), key=lambda x: x[1])
    for fname, rank in ranked[:10]:
        print(f"    {rank:2d}. {fname}")
    # Note: RFECV and sklearn.metrics are now imported at module scope (top of file)

    # ── Step 5b: Deep Learning on Raw CIR ────────────────────────────
    # Train CNN+Transformer directly on the raw 1016-sample CIR waveforms
    # (original 42K balanced dataset, no two-path expansion needed)
    print("\n" + "=" * 60)
    print("STEP 5b: CNN+Transformer on Raw CIR")
    print("=" * 60)

    # Prepare DL input: raw CIR waveforms + scaled scalar features
    cir_cols = [c for c in df.columns if c.startswith(
        'CIR') and c != 'CIR_PWR']
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
    # Add DL results to classification results dict for unified visualization
    cls_results['CNN+Transformer'] = dl_result

    # ── Step 5c: Synthetic Data Augmentation ────────────────────────
    # Test whether synthetic data improves classification robustness.
    # This addresses Data Preparation requirement VI.
    print("\n" + "=" * 60)
    print("STEP 5c: Synthetic Data Augmentation Experiment")
    print("=" * 60)

    # --- SMOTE on ML feature vectors ---
    print("\n>> SMOTE augmentation for ML classifiers")
    X_smote, y_smote, n_smote = apply_smote(
        X_train_cls, y_train_cls, target_ratio=SMOTE_TARGET_RATIO)

    # Re-train the best ML model (Random Forest) with SMOTE-augmented data
    rf_smote = RandomForestClassifier(
        n_estimators=RF_SMOTE_N_ESTIMATORS, max_depth=None,
        class_weight='balanced', random_state=RANDOM_STATE,
    )
    rf_smote.fit(X_smote, y_smote)
    y_pred_smote = rf_smote.predict(X_test_cls)
    y_prob_smote = rf_smote.predict_proba(X_test_cls)[:, 1]
    acc_smote = accuracy_score(y_test_cls, y_pred_smote)
    fpr_s, tpr_s, _ = roc_curve(y_test_cls, y_prob_smote)
    auc_smote = sk_auc(fpr_s, tpr_s)
    print(f"  RF + SMOTE: Accuracy={acc_smote:.4f}, AUC={auc_smote:.4f}")
    print(f"  RF original: Accuracy={cls_results['Random Forest']['accuracy']:.4f}, "
          f"AUC={cls_results['Random Forest']['auc']:.4f}")
    delta_acc = acc_smote - cls_results['Random Forest']['accuracy']
    delta_auc = auc_smote - cls_results['Random Forest']['auc']
    print(f"  Delta: Accuracy={delta_acc:+.4f}, AUC={delta_auc:+.4f}")

    # --- CIR augmentation for DL model ---
    print("\n>> CIR waveform augmentation for CNN+Transformer")
    cir_aug, scalar_aug, label_aug, n_cir_synth = generate_augmented_cir(
        X_cir_train, X_scalar_train, y_dl_train,
        augmentation_factor=CIR_AUG_FACTOR, noise_level=CIR_NOISE_LEVEL,
        max_shift=CIR_MAX_SHIFT, scale_range=CIR_SCALE_RANGE,
        random_state=RANDOM_STATE,
    )

    dl_result_aug = train_dl_classifier(
        cir_aug, scalar_aug, label_aug,
        X_cir_test, X_scalar_test, y_dl_test,
    )
    print(f"\n  CNN+Transformer + Augmentation: "
          f"Accuracy={dl_result_aug['accuracy']:.4f}, AUC={dl_result_aug['auc']:.4f}")
    print(f"  CNN+Transformer original: "
          f"Accuracy={dl_result['accuracy']:.4f}, AUC={dl_result['auc']:.4f}")
    delta_acc_dl = dl_result_aug['accuracy'] - dl_result['accuracy']
    delta_auc_dl = dl_result_aug['auc'] - dl_result['auc']
    print(f"  Delta: Accuracy={delta_acc_dl:+.4f}, AUC={delta_auc_dl:+.4f}")

    # ── Step 5d: Unsupervised Clustering Baseline ──────────────────
    print("\n" + "=" * 60)
    print("STEP 5d: Unsupervised Clustering Baseline")
    print("=" * 60)
    cluster_results = run_kmeans_analysis(
        X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    elbow_results = run_elbow_silhouette_analysis(X_train_cls)
    dbscan_results = run_dbscan_analysis(X_train_cls, X_test_cls, y_test_cls)

    # ── Step 5e: Ensemble Stacking ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5e: Ensemble Stacking")
    print("=" * 60)
    ensemble_results = build_ensemble(
        cls_results, X_train_cls, y_train_cls, X_test_cls, y_test_cls,
    )
    cls_results.update(ensemble_results)

    # ── Step 6: Distance Estimation ──────────────────────────────────
    # Train regressors separately for Path 1 and Path 2 range prediction
    print("\n" + "=" * 60)
    print("STEP 6: Distance Estimation")
    print("=" * 60)

    # Path 1 regression: use Path 1 features from the expanded dataset
    p1_mask_train = exp_train_idx[exp_train_idx < n_orig]
    p1_mask_test = exp_test_idx[exp_test_idx < n_orig]
    X_train_p1 = features_df.iloc[p1_mask_train].values
    y_train_p1 = labels_range[p1_mask_train]
    X_test_p1 = features_df.iloc[p1_mask_test].values
    y_test_p1 = labels_range[p1_mask_test]

    print("\n>> Path 1 Distance Estimation")
    reg_results_p1 = train_regressors(
        X_train_p1, y_train_p1, X_test_p1, y_test_p1, "Path 1")

    # Path 2 regression: use Path 2 features + original RANGE as extra feature
    # (Hint from spec: use FP_IDX and measured range to correlate to second path)
    p2_mask_train = exp_train_idx[exp_train_idx >= n_orig] - n_orig
    p2_mask_test = exp_test_idx[exp_test_idx >= n_orig] - n_orig

    # Add original RANGE as an additional feature for Path 2 estimation
    X_train_p2_base = features_df.iloc[p2_mask_train + n_orig].values
    X_test_p2_base = features_df.iloc[p2_mask_test + n_orig].values
    range_train = df['RANGE'].values[p2_mask_train].reshape(-1, 1)
    range_test = df['RANGE'].values[p2_mask_test].reshape(-1, 1)
    X_train_p2 = np.hstack([X_train_p2_base, range_train])
    X_test_p2 = np.hstack([X_test_p2_base, range_test])
    y_train_p2 = labels_range[p2_mask_train + n_orig]
    y_test_p2 = labels_range[p2_mask_test + n_orig]

    # Filter out samples where Path 2 was not detected (amplitude=0)
    # since their range labels would be unreliable
    valid_train = path2_amp[p2_mask_train] > 0
    valid_test = path2_amp[p2_mask_test] > 0
    print(
        f"\n  Path 2 valid samples: train={valid_train.sum()}, test={valid_test.sum()}")

    print("\n>> Path 2 Distance Estimation")
    reg_results_p2 = train_regressors(
        X_train_p2[valid_train], y_train_p2[valid_train],
        X_test_p2[valid_test], y_test_p2[valid_test], "Path 2",
    )

    # ── Step 7: Visualization ────────────────────────────────────────
    # Generate all plots organized by the 3D analytics stages
    print("\n" + "=" * 60)
    print("STEP 7: Generating Visualizations")
    print("=" * 60)

    # --- Data Preparation plots ---
    print("\n[Data Preparation]")
    viz.plot_class_distribution(df_raw['NLOS'].values, labels_cls)
    viz.plot_feature_distributions(df_raw)
    viz.plot_correlation_heatmap(df_raw)
    viz.plot_cir_examples(df, path1_idx, path1_amp, path2_idx, path2_amp)
    viz.plot_fp_idx_distribution(df_raw)

    # --- Data Mining plots ---
    print("\n[Data Mining]")
    if 'feature_importances' in cls_results.get('Random Forest', {}):
        viz.plot_feature_importance(
            cls_results['Random Forest']['feature_importances'],
            feature_names,
        )
    viz.plot_confusion_matrices(cls_results, y_test_cls)
    viz.plot_roc_curves(cls_results)
    viz.plot_pr_curves(cls_results, y_test_cls)
    viz.plot_model_comparison(cls_results)
    viz.plot_per_environment_heatmap(cls_results, X_test_cls, y_test_cls, env_ids_test)
    viz.plot_shap_summary(cls_results, X_test_cls, feature_names)
    viz.plot_rfe_curve(rfe_results)

    viz.plot_clustering(cluster_results, y_test_cls)
    viz.plot_elbow_silhouette(elbow_results)
    viz.plot_tsne_embedding(X_test_cls, y_test_cls, cluster_results['test_labels'])
    viz.plot_dbscan(dbscan_results, y_test_cls)

    # Plot transformer attention maps if DL model was trained
    if 'CNN+Transformer' in cls_results:
        viz.plot_attention_map(
            cls_results['CNN+Transformer']['model'],
            df_scaled, train_idx, test_idx,
        )

    # --- Results / Analysis plots ---
    print("\n[Results]")
    viz.plot_predicted_vs_actual(reg_results_p1, y_test_p1,
                                 reg_results_p2, y_test_p2[valid_test])
    viz.plot_residuals(reg_results_p1, y_test_p1,
                       reg_results_p2, y_test_p2[valid_test])
    viz.plot_regression_comparison(reg_results_p1, reg_results_p2)

    # --- Augmentation impact comparison ---
    print("\n[Augmentation Impact]")
    original_metrics = {
        'rf_acc': cls_results['Random Forest']['accuracy'],
        'rf_auc': cls_results['Random Forest']['auc'],
        'dl_acc': dl_result['accuracy'],
        'dl_auc': dl_result['auc'],
    }
    augmented_metrics = {
        'rf_acc': acc_smote,
        'rf_auc': auc_smote,
        'dl_acc': dl_result_aug['accuracy'],
        'dl_auc': dl_result_aug['auc'],
    }
    viz.plot_augmentation_impact(original_metrics, augmented_metrics)

    viz.plot_annotated_cir(df, path1_idx, path1_amp, path2_idx, path2_amp,
                           cls_results, features_df)

    # ── Cross-Pipeline Reference Evaluation ─────────────────────────
    # NOTE: This is NOT a strictly controlled benchmark. ML models receive
    # the 25-dim hand-crafted feature vector while the DL model receives the
    # raw 1016-sample CIR plus 11 scalar features. The two are evaluated on
    # the same underlying samples (the original 50/50 balanced Path-1 test
    # split), but the input representations differ — so results should be
    # read as cross-pipeline reference points rather than an apples-to-apples
    # comparison. This framing matches the discussion in the IEEE report.
    print("\n" + "=" * 60)
    print("CROSS-PIPELINE REFERENCE EVALUATION (balanced Path-1 test set)")
    print("  (different input representations — not a controlled benchmark)")
    print("=" * 60)
    # Evaluate ML models on the original balanced test set (path1 only, 50/50 class balance)
    X_test_fair = features_df.iloc[test_idx].values
    y_test_fair = labels_cls[test_idx]
    print(f"  Balanced test set: {len(y_test_fair)} samples "
          f"(LOS={int((y_test_fair == 0).sum())}, NLOS={int((y_test_fair == 1).sum())})")

    # acc_fn/roc_fn/auc_fn are aliased at module scope (top of file)
    # Evaluate individual ML models on balanced test set (skip ensembles — handled separately below)
    ml_model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosted Trees',
                      'XGBoost']
    for name in ml_model_names:
        if name not in cls_results or 'model' not in cls_results[name]:
            continue
        model = cls_results[name]['model']
        y_prob_fair = model.predict_proba(X_test_fair)[:, 1]
        y_pred_fair = (y_prob_fair >= 0.5).astype(int)
        acc_fair = acc_fn(y_test_fair, y_pred_fair)
        fpr_f, tpr_f, _ = roc_fn(y_test_fair, y_prob_fair)
        auc_fair = auc_fn(fpr_f, tpr_f)
        print(f"  {name:30s}: Accuracy={acc_fair:.4f}, AUC={auc_fair:.4f}")

    # DL model on its own balanced test set (already evaluated)
    # dl_res is initialised here so downstream fusion can reference it safely
    # even if the CNN+Transformer block failed earlier (we skip fusion in that case).
    dl_res = cls_results.get('CNN+Transformer')
    if dl_res is not None:
        print(
            f"  {'CNN+Transformer':30s}: Accuracy={dl_res['accuracy']:.4f}, AUC={dl_res['auc']:.4f}")

    # ── DL+ML Ensemble Fusion ─────────────────────────────────────
    # Average best ML ensemble probabilities with DL probabilities on balanced test set.
    # NOTE: the DL probabilities used here come from the ORIGINAL (non-augmented)
    # CNN+Transformer run. The augmented DL model is not re-evaluated on the
    # fair test set because its training set was synthetically expanded and
    # its stored y_prob is on the same balanced test set anyway.
    best_ml_names = [n for n in ['Random Forest', 'Gradient Boosted Trees', 'XGBoost']
                     if n in cls_results and 'model' in cls_results[n]]
    if dl_res is not None and len(best_ml_names) >= 2:
        ml_fair_probs = np.column_stack([
            cls_results[n]['model'].predict_proba(X_test_fair)[:, 1]
            for n in best_ml_names
        ])
        ml_ens_prob = ml_fair_probs.mean(axis=1)
        # DL already evaluated on balanced test set (original, non-augmented model)
        dl_prob_fair = dl_res['y_prob']
        combined_prob = 0.5 * ml_ens_prob + 0.5 * dl_prob_fair
        combined_pred = (combined_prob >= 0.5).astype(int)
        acc_combined = acc_fn(y_test_fair, combined_pred)
        fpr_c, tpr_c, _ = roc_fn(y_test_fair, combined_prob)
        auc_combined = auc_fn(fpr_c, tpr_c)
        print(
            f"  {'DL+ML Fusion (non-aug DL)':30s}: Accuracy={acc_combined:.4f}, AUC={auc_combined:.4f}")

    # Ensemble models: re-evaluate on fair test set
    base_names_fair = [n for n in ['Random Forest', 'Gradient Boosted Trees', 'XGBoost']
                       if n in cls_results and 'model' in cls_results[n]]
    if len(base_names_fair) >= 2:
        fair_probs = np.column_stack([
            cls_results[n]['model'].predict_proba(X_test_fair)[:, 1]
            for n in base_names_fair
        ])
        for ens_name in ['Ensemble (Average)', 'Ensemble (Stacked)']:
            if ens_name not in cls_results:
                continue
            if ens_name == 'Ensemble (Average)':
                ens_prob = fair_probs.mean(axis=1)
            else:
                # Use the stored meta-learner for proper stacked prediction
                meta_model = cls_results[ens_name].get('model')
                if meta_model is not None:
                    ens_prob = meta_model.predict_proba(fair_probs)[:, 1]
                else:
                    ens_prob = fair_probs.mean(axis=1)
            ens_pred = (ens_prob >= 0.5).astype(int)
            acc_ens = acc_fn(y_test_fair, ens_pred)
            fpr_e, tpr_e, _ = roc_fn(y_test_fair, ens_prob)
            auc_ens = auc_fn(fpr_e, tpr_e)
            print(
                f"  {ens_name:30s}: Accuracy={acc_ens:.4f}, AUC={auc_ens:.4f} (fair)")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nClassification Results:")
    for name, res in cls_results.items():
        print(
            f"  {name}: Accuracy={res['accuracy']:.4f}, AUC={res['auc']:.4f}")

    print("\nPath 1 Distance Estimation:")
    for name, res in reg_results_p1.items():
        print(f"  {name}: RMSE={res['rmse']:.4f}m, R2={res['r2']:.4f}")

    print("\nPath 2 Distance Estimation:")
    for name, res in reg_results_p2.items():
        print(f"  {name}: RMSE={res['rmse']:.4f}m, R2={res['r2']:.4f}")

    print("\nUnsupervised Clustering Baseline:")
    print(f"  K-Means (k=2): Accuracy={cluster_results['test_accuracy']:.4f}, "
          f"Silhouette={cluster_results['silhouette_test']:.4f}, "
          f"ARI={cluster_results['ari']:.4f}")

    print("\nSynthetic Data Augmentation Impact:")
    print(f"  RF + SMOTE:                Acc={acc_smote:.4f} (delta={delta_acc:+.4f}), "
          f"AUC={auc_smote:.4f} (delta={delta_auc:+.4f})")
    print(f"  CNN+Transformer + CIR Aug: Acc={dl_result_aug['accuracy']:.4f} "
          f"(delta={delta_acc_dl:+.4f}), AUC={dl_result_aug['auc']:.4f} "
          f"(delta={delta_auc_dl:+.4f})")

    if ensemble_results:
        print("\nEnsemble Results:")
        for name, res in ensemble_results.items():
            print(
                f"  {name}: Accuracy={res['accuracy']:.4f}, AUC={res['auc']:.4f}")

    print(f"\nAll plots saved to {viz.PLOT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
