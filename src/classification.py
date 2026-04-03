"""
classification.py — LOS/NLOS classification models with evaluation.

This module implements the Data Mining stage for the classification task:
  - Logistic Regression: Linear baseline with balanced class weights.
  - Random Forest: Ensemble bagging classifier with GridSearchCV hyperparameter
    tuning over n_estimators and max_depth.
  - Histogram Gradient Boosted Trees: Fast histogram-based boosting with
    GridSearchCV tuning over learning_rate, max_iter, and max_depth.
  - XGBoost: Gradient boosting with GridSearchCV (if installed).

All models use 3-fold stratified cross-validation for hyperparameter selection.

Evaluation metrics (Data Analysis stage):
  - Accuracy, AUC, Confusion Matrix, Classification Report

Libraries: sklearn (models, GridSearchCV, metrics), numpy
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _xgb_device():
    """Auto-detect best XGBoost device: 'cuda' if available, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def train_classifiers(X_train, y_train, X_test, y_test):
    """
    Train ML classifiers and evaluate on the test set.

    Parameters:
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.

    Returns:
        dict: {model_name: {model, y_pred, y_prob, accuracy, auc, ...}}
    """
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 1. Logistic Regression — linear baseline ─────────────────────
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = _evaluate(lr, X_test, y_test)

    # ── 2. SVM with RBF Kernel and GridSearchCV ──────────────────────
    # SVM is O(n²)-O(n³), so subsample training data for tractability.
    print("\n--- SVM (RBF) (GridSearchCV, subsampled) ---")
    svm_max_samples = min(15000, len(X_train))
    rng = np.random.RandomState(42)
    svm_idx = rng.choice(len(X_train), svm_max_samples, replace=False)
    X_train_svm, y_train_svm = X_train[svm_idx], y_train[svm_idx]
    print(f"  Subsampled to {svm_max_samples} training samples for SVM")

    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    param_grid_svm = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
    }
    gs_svm = GridSearchCV(
        svm, param_grid_svm, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1
    )
    gs_svm.fit(X_train_svm, y_train_svm)
    print(f"  Best params: {gs_svm.best_params_}")
    results["SVM (RBF)"] = _evaluate(gs_svm.best_estimator_, X_test, y_test)

    # ── 3. Random Forest with GridSearchCV ───────────────────────────
    print("\n--- Random Forest (GridSearchCV) ---")
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [20, None],
        "min_samples_leaf": [1, 3],
    }
    gs = GridSearchCV(
        rf, param_grid, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"  Best params: {gs.best_params_}")
    results["Random Forest"] = _evaluate(gs.best_estimator_, X_test, y_test)
    results["Random Forest"]["feature_importances"] = (
        gs.best_estimator_.feature_importances_
    )

    # ── 3. Histogram Gradient Boosted Trees with GridSearchCV ─────────
    # HistGradientBoosting is 10-50x faster than GradientBoosting for
    # large datasets — uses histogram binning like LightGBM.
    print("\n--- Gradient Boosted Trees (GridSearchCV) ---")
    # Compute sample weights for class imbalance (HistGBT doesn't support class_weight)
    class_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    sample_weight = np.where(y_train == 0, 1.0 / class_ratio, 1.0)
    sample_weight = sample_weight / sample_weight.mean()  # normalize

    gbt = HistGradientBoostingClassifier(random_state=42)
    param_grid_gbt = {
        "learning_rate": [0.05, 0.1],
        "max_iter": [200, 500],
        "max_depth": [4, 6],
    }
    gs_gbt = GridSearchCV(
        gbt, param_grid_gbt, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1
    )
    gs_gbt.fit(X_train, y_train, sample_weight=sample_weight)
    print(f"  Best params: {gs_gbt.best_params_}")
    results["Gradient Boosted Trees"] = _evaluate(
        gs_gbt.best_estimator_, X_test, y_test
    )

    # ── 4. XGBoost with GridSearchCV ──────────────────────────────────
    if HAS_XGB:
        print("\n--- XGBoost (GridSearchCV) ---")
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos = n_neg / max(n_pos, 1)
        xgb = XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos,
            device=_xgb_device(),
        )
        param_grid_xgb = {
            "n_estimators": [200, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
        gs_xgb = GridSearchCV(
            xgb, param_grid_xgb, scoring="f1_weighted", cv=cv, n_jobs=1, verbose=1
        )
        gs_xgb.fit(X_train, y_train)
        print(f"  Best params: {gs_xgb.best_params_}")
        results["XGBoost"] = _evaluate(gs_xgb.best_estimator_, X_test, y_test)
        results["XGBoost"]["feature_importances"] = (
            gs_xgb.best_estimator_.feature_importances_
        )
    else:
        print("\n--- XGBoost: skipped (xgboost not installed) ---")

    return results


def _evaluate(model, X_test, y_test):
    """Evaluate a trained classifier and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["LOS", "NLOS"])
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"  Accuracy: {acc:.4f}, AUC: {roc_auc:.4f}")
    print(report)

    return {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "auc": roc_auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "report": report,
    }
