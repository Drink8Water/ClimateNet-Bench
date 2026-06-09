"""Model training helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from climatenet.training.metrics import regression_metrics
from climatenet.training.validation import ValidationSplit


def collect_feature_importance(
    model_name: str,
    model: Any,
    feature_names: list[str],
    split: ValidationSplit,
) -> pd.DataFrame:
    """Collect feature importance or absolute linear coefficients."""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "importance"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)
        importance_type = "absolute_coefficient"
    else:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "validation_strategy": split.strategy,
            "train_region": split.train_region,
            "test_region": split.test_region,
            "train_period": split.train_period,
            "test_period": split.test_period,
            "model": model_name,
            "feature": feature_names,
            "importance": values,
            "importance_type": importance_type,
        }
    )


def train_on_split(
    split: ValidationSplit,
    models: dict[str, Any],
    feature_names: list[str],
    target_column: str,
    feature_set_name: str = "default",
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    """Train all models for a validation split."""
    x_train = split.train_data[feature_names]
    y_train = split.train_data[target_column]
    x_test = split.test_data[feature_names]
    y_test = split.test_data[target_column]

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = regression_metrics(y_test, predictions)

        metrics_rows.append(
            {
                "validation_strategy": split.strategy,
                "train_region": split.train_region,
                "test_region": split.test_region,
                "train_period": split.train_period,
                "test_period": split.test_period,
                "feature_set": feature_set_name,
                "model": model_name,
                "model_name": model_name,
                "n_train": len(split.train_data),
                "n_test": len(split.test_data),
                **metrics,
            }
        )

        prediction_data = split.test_data[["region", "year", "month", "latitude", "longitude"]].copy()
        prediction_data["validation_strategy"] = split.strategy
        prediction_data["train_region"] = split.train_region
        prediction_data["test_region"] = split.test_region
        prediction_data["train_period"] = split.train_period
        prediction_data["test_period"] = split.test_period
        prediction_data["feature_set"] = feature_set_name
        prediction_data["model"] = model_name
        prediction_data["model_name"] = model_name
        prediction_data["actual"] = y_test.to_numpy()
        prediction_data["prediction"] = predictions
        prediction_frames.append(prediction_data)

        importance_frames.append(collect_feature_importance(model_name, model, feature_names, split))

    return metrics_rows, prediction_frames, importance_frames
