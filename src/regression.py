"""Distance estimation models for path 1 and path 2."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_regressors(X_train, y_train, X_test, y_test, path_name=""):
    """
    Train Ridge, RF, and GBT regressors for distance estimation.

    Returns dict of {name: {model, y_pred, metrics}}.
    """
    results = {}

    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1,
        ),
        'Gradient Boosted Regressor': GradientBoostingRegressor(
            n_estimators=200, random_state=42,
        ),
    }

    for name, model in models.items():
        print(f"\n--- {name} ({path_name}) ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  RMSE: {rmse:.4f} m, MAE: {mae:.4f} m, R²: {r2:.4f}")

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }

    return results
