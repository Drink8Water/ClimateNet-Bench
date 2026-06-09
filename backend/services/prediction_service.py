"""Prediction and residual data access from experiment CSV files."""

from __future__ import annotations

import pandas as pd

from backend.config import DEFAULT_PREDICTION_LIMIT, MAX_PREDICTION_LIMIT, get_experiment_dir
from backend.data_loader import read_csv_cached


def get_predictions(
    experiment_id: str,
    model: str | None = None,
    validation_strategy: str | None = None,
    region: str | None = None,
    feature_set: str | None = None,
    year: int | None = None,
    month: int | None = None,
    limit: int = DEFAULT_PREDICTION_LIMIT,
) -> list[dict]:
    """Return sampled prediction rows from an experiment."""
    exp_dir = get_experiment_dir(experiment_id)
    path = exp_dir / "predictions.csv"
    if not path.exists():
        return []

    limit = min(limit, MAX_PREDICTION_LIMIT)
    df = pd.read_csv(path, nrows=limit + 1) if limit else pd.read_csv(path)

    if model:
        # Match either 'model' or 'model_name' column
        col = "model" if "model" in df.columns else "model_name"
        df = df[df[col] == model]
    if validation_strategy:
        df = df[df["validation_strategy"] == validation_strategy]
    if region:
        df = df[df["region"] == region]
    if feature_set:
        df = df[df["feature_set"] == feature_set]
    if year:
        df = df[df["year"] == year]
    if month:
        df = df[df["month"] == month]

    df = df.head(limit)
    # Normalize column names: ensure 'model' and actual/prediction exist
    if "model_name" in df.columns and "model" not in df.columns:
        df = df.rename(columns={"model_name": "model"})
    if "y_true" in df.columns:
        df = df.rename(columns={"y_true": "actual", "y_pred": "prediction"})

    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def get_residuals(
    experiment_id: str,
    region: str | None = None,
    limit: int = DEFAULT_PREDICTION_LIMIT,
) -> list[dict]:
    """Return prediction rows with computed residuals."""
    predictions = get_predictions(experiment_id, region=region, limit=limit)
    results = []
    for row in predictions:
        actual = row.get("actual")
        pred = row.get("prediction")
        residual = actual - pred if (actual is not None and pred is not None) else None
        results.append({
            "year": row.get("year"),
            "month": row.get("month"),
            "region": row.get("region"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "actual": actual,
            "prediction": pred,
            "residual": residual,
        })
    return results


def get_prediction_summary(experiment_id: str, region: str | None = None) -> dict:
    """Compute summary statistics for predictions."""
    residuals = get_residuals(experiment_id, region=region, limit=MAX_PREDICTION_LIMIT)
    residual_values = [r["residual"] for r in residuals if r["residual"] is not None]
    if not residual_values:
        return {
            "mean_residual": 0.0,
            "residual_std": 0.0,
            "max_absolute_error": 0.0,
            "prediction_count": 0,
        }
    import numpy as np

    arr = np.array(residual_values)
    return {
        "mean_residual": float(np.mean(arr)),
        "residual_std": float(np.std(arr)),
        "max_absolute_error": float(np.max(np.abs(arr))),
        "prediction_count": len(residual_values),
    }
