"""
regression.py — Distance estimation models for Path 1 and Path 2.

This module implements the Data Mining stage for the distance estimation task.
The goal is to predict the physical range (in metres) for each of the two
detected propagation paths, enabling accurate UWB indoor positioning.

Three regression algorithms are compared:
  - Ridge Regression: L2-regularised linear regression (baseline).
  - Random Forest Regressor: Ensemble of decision trees for non-linear regression.
  - Gradient Boosted Regressor: Sequential boosting ensemble for non-linear regression.

Evaluation metrics (Data Analysis stage):
  - RMSE (Root Mean Squared Error): Average prediction error in metres,
    penalises large errors more heavily.
  - MAE (Mean Absolute Error): Average absolute prediction error in metres.
  - R² (Coefficient of Determination): Fraction of variance explained by the model,
    where 1.0 = perfect and 0.0 = predicting the mean.

Libraries: sklearn (Ridge, RandomForestRegressor, GradientBoostingRegressor, metrics), numpy
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_regressors(X_train, y_train, X_test, y_test, path_name=""):
    """
    Train three regression models for distance estimation on a single path.

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

    # Define the three regression models to compare
    models = {
        # Ridge Regression: L2-regularised linear model (alpha controls regularisation).
        # Provides a linear baseline for distance estimation.
        'Ridge Regression': Ridge(alpha=1.0),

        # Random Forest Regressor: Ensemble of 200 decision trees (bagging).
        # Captures non-linear relationships between features and range.
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1,
        ),

        # Gradient Boosted Regressor: Sequential ensemble of 200 boosted trees.
        # Often achieves best performance via iterative error correction.
        'Gradient Boosted Regressor': GradientBoostingRegressor(
            n_estimators=200, random_state=42,
        ),
    }

    for name, model in models.items():
        print(f"\n--- {name} ({path_name}) ---")

        # Fit the model on training data
        model.fit(X_train, y_train)

        # Predict distances on test set
        y_pred = model.predict(X_test)

        # Compute regression metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root mean squared error
        mae = mean_absolute_error(y_test, y_pred)            # Mean absolute error
        r2 = r2_score(y_test, y_pred)                        # Coefficient of determination

        print(f"  RMSE: {rmse:.4f} m, MAE: {mae:.4f} m, R²: {r2:.4f}")

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }

    return results
