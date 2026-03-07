"""
ensemble.py — Ensemble stacking of ML classifiers for improved classification.

Combines predictions from multiple ML models (RF, GBT, XGBoost) via:
  1. Simple averaging of predicted probabilities
  2. Stacked generalization with a logistic regression meta-learner

Uses cross_val_predict for train-set probabilities to avoid data leakage.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc,
)


def build_ensemble(cls_results, X_train, y_train, X_test, y_test):
    """
    Build ensemble predictions from trained ML classifiers.

    Parameters:
        cls_results (dict): Results dict from train_classifiers(), keyed by model name.
        X_train (np.ndarray): Training features for cross-validated stacking.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        dict: {method_name: results_dict} for 'Ensemble (Average)' and 'Ensemble (Stacked)'.
    """
    # Collect base models that have trained sklearn estimators
    base_names = [n for n in ['Random Forest', 'Gradient Boosted Trees', 'XGBoost']
                  if n in cls_results and 'model' in cls_results[n]]
    if len(base_names) < 2:
        print("  Ensemble: need at least 2 base models, skipping.")
        return {}

    base_models = [cls_results[n]['model'] for n in base_names]
    print(f"\n--- Ensemble ({len(base_names)} models: {', '.join(base_names)}) ---")

    # ── Simple average ensemble ──────────────────────────────────────
    test_probs = np.column_stack([cls_results[n]['y_prob'] for n in base_names])
    avg_prob = test_probs.mean(axis=1)
    avg_pred = (avg_prob >= 0.5).astype(int)

    avg_results = _compute_metrics(y_test, avg_pred, avg_prob, "Ensemble (Average)")

    # ── Stacked ensemble with logistic regression meta-learner ───────
    # Generate train-set probabilities via cross_val_predict to avoid leakage
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_probs_list = []
    for model in base_models:
        probs = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')
        train_probs_list.append(probs[:, 1])
    train_meta = np.column_stack(train_probs_list)

    # Train meta-learner on cross-validated probabilities
    meta = LogisticRegression(random_state=42, max_iter=1000)
    meta.fit(train_meta, y_train)

    # Predict on test set using actual model probabilities
    test_meta = test_probs  # already [n_test, n_models]
    stack_prob = meta.predict_proba(test_meta)[:, 1]
    stack_pred = (stack_prob >= 0.5).astype(int)

    stack_results = _compute_metrics(y_test, stack_pred, stack_prob, "Ensemble (Stacked)")

    return {
        'Ensemble (Average)': avg_results,
        'Ensemble (Stacked)': stack_results,
    }


def _compute_metrics(y_test, y_pred, y_prob, name):
    """Compute standard classification metrics and print summary."""
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['LOS', 'NLOS'])
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"  {name}: Accuracy={acc:.4f}, AUC={roc_auc:.4f}")
    print(report)

    return {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'report': report,
    }
