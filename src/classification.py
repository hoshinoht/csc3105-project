"""LOS/NLOS classification models with evaluation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc,
)


def train_classifiers(X_train, y_train, X_test, y_test):
    """
    Train Logistic Regression, Random Forest, and GBT classifiers.

    Returns dict of {name: {model, y_pred, y_prob, metrics}}.
    """
    results = {}

    # 1. Logistic Regression baseline
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results['Logistic Regression'] = _evaluate(lr, X_test, y_test)

    # 2. Random Forest with GridSearch
    print("\n--- Random Forest (GridSearchCV) ---")
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [20, None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(rf, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"  Best params: {gs.best_params_}")
    results['Random Forest'] = _evaluate(gs.best_estimator_, X_test, y_test)
    results['Random Forest']['feature_importances'] = gs.best_estimator_.feature_importances_

    # 3. Gradient Boosted Trees
    print("\n--- Gradient Boosted Trees (GridSearchCV) ---")
    gbt = GradientBoostingClassifier(random_state=42)
    param_grid_gbt = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 300],
        'max_depth': [3, 6],
    }
    gs_gbt = GridSearchCV(gbt, param_grid_gbt, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1)
    gs_gbt.fit(X_train, y_train)
    print(f"  Best params: {gs_gbt.best_params_}")
    results['Gradient Boosted Trees'] = _evaluate(gs_gbt.best_estimator_, X_test, y_test)

    return results


def _evaluate(model, X_test, y_test):
    """Evaluate a classifier and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['LOS', 'NLOS'])
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

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
