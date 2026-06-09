"""Evaluate model predictions using regression metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import METRICS_PATH, PREDICTIONS_PATH, ensure_directories


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate MAE, RMSE, and R2 for regression predictions."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Evaluate every model prediction column in the predictions file."""
    if "actual" not in predictions.columns:
        raise ValueError("predictions.csv must contain an 'actual' column.")

    metrics = {}
    prediction_columns = [column for column in predictions.columns if column.endswith("_prediction")]

    for column in prediction_columns:
        model_name = column.replace("_prediction", "")
        metrics[model_name] = calculate_metrics(predictions["actual"], predictions[column])

    return metrics


def main() -> None:
    """Load predictions, calculate metrics, and save JSON output."""
    ensure_directories()
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions at {PREDICTIONS_PATH}. Run: python src/train.py")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    metrics = evaluate_predictions(predictions)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved model metrics to {METRICS_PATH}")
    for model_name, model_metrics in metrics.items():
        print(
            f"{model_name}: "
            f"MAE={model_metrics['MAE']:.3f}, "
            f"RMSE={model_metrics['RMSE']:.3f}, "
            f"R2={model_metrics['R2']:.3f}"
        )


if __name__ == "__main__":
    main()

