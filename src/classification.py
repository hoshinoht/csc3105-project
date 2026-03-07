"""
classification.py — LOS/NLOS classification models with evaluation.

This module implements the Data Mining stage for the classification task:
  - Logistic Regression: Linear baseline with balanced class weights.
  - Random Forest: Ensemble bagging classifier with GridSearchCV hyperparameter
    tuning over n_estimators and max_depth.
  - Gradient Boosted Trees: Ensemble boosting classifier with GridSearchCV tuning
    over learning_rate, n_estimators, and max_depth.

All models use 5-fold stratified cross-validation for hyperparameter selection,
ensuring each fold preserves the LOS/NLOS class ratio.

Evaluation metrics (Data Analysis stage):
  - Accuracy: Overall correct classification rate
  - AUC (Area Under ROC Curve): Ranking quality, threshold-independent
  - Confusion Matrix: True/false positive/negative breakdown
  - Classification Report: Per-class precision, recall, and F1-score

Libraries: sklearn (models, GridSearchCV, metrics), numpy
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc,
)


def train_classifiers(X_train, y_train, X_test, y_test):
    """
    Train three ML classifiers and evaluate on the test set.

    Models trained:
      1. Logistic Regression — linear baseline, fast, interpretable coefficients.
         Uses class_weight='balanced' to handle the 25/75 LOS/NLOS imbalance
         in the two-path expanded dataset.
      2. Random Forest — ensemble of decision trees (bagging). GridSearchCV
         searches over n_estimators=[100,300] and max_depth=[20,None].
      3. Gradient Boosted Trees — sequential boosting ensemble. GridSearchCV
         searches over learning_rate=[0.01,0.1], n_estimators=[100,300],
         max_depth=[3,6].

    Parameters:
        X_train (np.ndarray): Training feature matrix, shape (N_train, 18).
        y_train (np.ndarray): Training labels (0=LOS, 1=NLOS), shape (N_train,).
        X_test (np.ndarray): Test feature matrix, shape (N_test, 18).
        y_test (np.ndarray): Test labels, shape (N_test,).

    Returns:
        dict: {model_name: {model, y_pred, y_prob, accuracy, auc,
               confusion_matrix, fpr, tpr, report}} for each classifier.
    """
    results = {}

    # ── 1. Logistic Regression — linear baseline ─────────────────────
    # Fast to train, provides a performance floor. class_weight='balanced'
    # inversely weights classes by their frequency to handle imbalance.
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results['Logistic Regression'] = _evaluate(lr, X_test, y_test)

    # ── 2. Random Forest with GridSearchCV ───────────────────────────
    # Ensemble of decorrelated decision trees via bagging (bootstrap aggregation).
    # GridSearchCV performs exhaustive search over hyperparameter combinations
    # using 5-fold stratified CV to select the best configuration.
    print("\n--- Random Forest (GridSearchCV) ---")
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 300],   # Number of trees in the forest
        'max_depth': [20, None],       # Tree depth limit (None = unlimited)
    }
    # 5-fold stratified CV ensures each fold has the same LOS/NLOS ratio
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(rf, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"  Best params: {gs.best_params_}")
    results['Random Forest'] = _evaluate(gs.best_estimator_, X_test, y_test)
    # Store Gini-based feature importances for visualization (feature importance ranking)
    results['Random Forest']['feature_importances'] = gs.best_estimator_.feature_importances_

    # ── 3. Gradient Boosted Trees with GridSearchCV ──────────────────
    # Sequential ensemble: each tree corrects the errors of the previous one.
    # More parameters to tune than RF, but often achieves better performance.
    print("\n--- Gradient Boosted Trees (GridSearchCV) ---")
    gbt = GradientBoostingClassifier(random_state=42)
    param_grid_gbt = {
        'learning_rate': [0.01, 0.1],  # Step size shrinkage (regularization)
        'n_estimators': [100, 300],     # Number of boosting stages
        'max_depth': [3, 6],            # Depth of each individual tree
    }
    gs_gbt = GridSearchCV(gbt, param_grid_gbt, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1)
    gs_gbt.fit(X_train, y_train)
    print(f"  Best params: {gs_gbt.best_params_}")
    results['Gradient Boosted Trees'] = _evaluate(gs_gbt.best_estimator_, X_test, y_test)

    return results


def _evaluate(model, X_test, y_test):
    """
    Evaluate a trained classifier on the test set and compute all metrics.

    Metrics computed:
      - accuracy: fraction of correctly classified samples
      - y_prob: predicted probability of NLOS class (for ROC/AUC)
      - confusion_matrix: 2x2 matrix of TP/FP/TN/FN
      - fpr, tpr: false/true positive rates at varying thresholds (for ROC curve)
      - auc: area under the ROC curve (threshold-independent ranking quality)
      - report: per-class precision, recall, F1-score text summary

    Parameters:
        model: Trained sklearn classifier with predict() and predict_proba().
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): True test labels.

    Returns:
        dict: Dictionary containing the model, predictions, and all metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # P(NLOS) for ROC computation

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['LOS', 'NLOS'])
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)  # ROC curve at multiple thresholds
    roc_auc = auc(fpr, tpr)  # Integrate ROC curve for AUC score

    print(f"  Accuracy: {acc:.4f}, AUC: {roc_auc:.4f}")
    print(report)

    return {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'report': report,
    }
