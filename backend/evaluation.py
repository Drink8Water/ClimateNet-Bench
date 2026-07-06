"""Synchronous evaluation helpers for submitted prediction files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from climatenet.evaluation.detection import compute_event_detection_table
from climatenet.evaluation.hydroclimate_labels import ALL_EVENT_TYPES
from climatenet.evaluation.metrics import evaluate_regression


def evaluate_prediction_csv(prediction_csv_path: str) -> dict[str, float]:
    """Compute regression and optional hydroclimate event metrics."""
    path = Path(prediction_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_csv_path}")

    predictions = pd.read_csv(path)
    required_columns = {"actual", "prediction"}
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {sorted(missing)}")

    metric_values = evaluate_regression(
        predictions["actual"].to_numpy(),
        predictions["prediction"].to_numpy(),
    )

    available_event_types = [
        event_type
        for event_type in ALL_EVENT_TYPES
        if event_type in predictions.columns and f"{event_type}_pred" in predictions.columns
    ]
    if not available_event_types:
        return metric_values

    detection_input = predictions.rename(columns={"actual": "y_true", "prediction": "y_pred"})
    detection_table = compute_event_detection_table(detection_input, event_types=available_event_types)
    for row in detection_table.to_dict(orient="records"):
        event_type = row["event_type"]
        for metric_name in ("pod", "far", "csi", "intensity_bias"):
            metric_values[f"{event_type}_{metric_name}"] = row[metric_name]

    return metric_values
