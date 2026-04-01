"""
regression.py — Distance estimation models for Path 1 and Path 2.

This module implements the Data Mining stage for the distance estimation task.
The goal is to predict the physical range (in metres) for each of the two
detected propagation paths, enabling accurate UWB indoor positioning.

Four regression algorithms are compared:
  - Ridge Regression: L2-regularised linear regression (baseline).
  - Random Forest Regressor: Ensemble of decision trees for non-linear regression.
  - Gradient Boosted Regressor: Sequential boosting ensemble for non-linear regression.
  - XGBoost Regressor: GPU-accelerated gradient boosting (if available).

Evaluation metrics (Data Analysis stage):
  - RMSE (Root Mean Squared Error): Average prediction error in metres,
    penalises large errors more heavily.
  - MAE (Mean Absolute Error): Average absolute prediction error in metres.
  - R² (Coefficient of Determination): Fraction of variance explained by the model,
    where 1.0 = perfect and 0.0 = predicting the mean.

Libraries: sklearn (Ridge, RandomForestRegressor, GradientBoostingRegressor, metrics),
           xgboost (GPU-accelerated regression), numpy
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def train_regressors(X_train, y_train, X_test, y_test, path_name=""):
    """
    Train regression models for distance estimation on a single path.

    This function is called twice in the main pipeline:
      1. For Path 1 (first/direct path) — using 18 per-path features.
      2. For Path 2 (reflected/multipath) — using 18 features + RANGE as an
         additional feature (since Path 2's range correlates with Path 1's
         measured range plus the multipath delay offset).

    Parameters:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training distance labels in metres.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test distance labels in metres.
        path_name (str): Identifier for logging (e.g., "Path 1" or "Path 2").

    Returns:
        dict: {model_name: {model, y_pred, rmse, mae, r2}} for each regressor.
    """
    results = {}

    # ── 1. Ridge Regression — L2-regularised linear baseline ────────────
    print(f"\n--- Ridge Regression ({path_name}) ---")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results['Ridge Regression'] = _evaluate_regressor(ridge, X_test, y_test)

    # ── 2. Random Forest Regressor with GridSearchCV ────────────────────
    print(f"\n--- Random Forest Regressor (GridSearchCV) ({path_name}) ---")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [20, None],
        'min_samples_leaf': [1, 3],
    }
    gs_rf = GridSearchCV(rf, rf_param_grid, scoring='neg_mean_squared_error',
                         cv=3, n_jobs=-1, verbose=1)
    gs_rf.fit(X_train, y_train)
    print(f"  Best params: {gs_rf.best_params_}")
    results['Random Forest Regressor'] = _evaluate_regressor(
        gs_rf.best_estimator_, X_test, y_test)

    # ── 3. Gradient Boosted Regressor with GridSearchCV ─────────────────
    print(f"\n--- Gradient Boosted Regressor (GridSearchCV) ({path_name}) ---")
    gbr = GradientBoostingRegressor(random_state=42)
    gbr_param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 7],
    }
    gs_gbr = GridSearchCV(gbr, gbr_param_grid, scoring='neg_mean_squared_error',
                          cv=3, n_jobs=-1, verbose=1)
    gs_gbr.fit(X_train, y_train)
    print(f"  Best params: {gs_gbr.best_params_}")
    results['Gradient Boosted Regressor'] = _evaluate_regressor(
        gs_gbr.best_estimator_, X_test, y_test)

    # ── 4. XGBoost Regressor with GPU-accelerated GridSearchCV ──────────
    if HAS_XGB:
        print(f"\n--- XGBoost Regressor (GridSearchCV, GPU) ({path_name}) ---")
        xgb = XGBRegressor(
            random_state=42,
            device='cuda',
        )
        xgb_param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
        }
        gs_xgb = GridSearchCV(xgb, xgb_param_grid, scoring='neg_mean_squared_error',
                              cv=3, n_jobs=1, verbose=1)
        gs_xgb.fit(X_train, y_train)
        print(f"  Best params: {gs_xgb.best_params_}")
        results['XGBoost Regressor'] = _evaluate_regressor(
            gs_xgb.best_estimator_, X_test, y_test)
    else:
        print(f"\n--- XGBoost Regressor: skipped (xgboost not installed) ---")

    return results


def _evaluate_regressor(model, X_test, y_test):
    """Evaluate a trained regressor and return metrics dict."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  RMSE: {rmse:.4f} m, MAE: {mae:.4f} m, R²: {r2:.4f}")
    return {
        'model': model,
        'y_pred': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }
