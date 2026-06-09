"""Train models under random, spatial, and cross-region validation strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    ALL_METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURES_PATH,
    MODEL_FEATURES,
    PREDICTIONS_PATH,
    RANDOM_SEED,
    RANDOM_SPLIT_METRICS_PATH,
    REGION_TRANSFER_METRICS_PATH,
    SPATIAL_HOLDOUT_METRICS_PATH,
    TARGET_COLUMN,
    ensure_directories,
)
from validation import ValidationSplit, build_validation_splits

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("xgboost is required. Install dependencies with: pip install -r requirements.txt") from exc


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load engineered features with a clear error if they are missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing features at {path}. Run: python src/features.py")
    return pd.read_csv(path)


def build_models() -> dict[str, Any]:
    """Create the Phase 3 model set, skipping LightGBM if it is unavailable."""
    models: dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_SEED,
            min_samples_leaf=3,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
    }

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print("LightGBM is not installed; skipping lightgbm model.")
    else:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            verbose=-1,
        )

    return models


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R2 for regression predictions."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def collect_feature_importance(
    model_name: str,
    model: Any,
    validation_split: ValidationSplit,
) -> pd.DataFrame:
    """Collect model feature importance or absolute linear coefficients."""
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
            "validation_strategy": validation_split.strategy,
            "train_region": validation_split.train_region,
            "test_region": validation_split.test_region,
            "model": model_name,
            "feature": MODEL_FEATURES,
            "importance": values,
            "importance_type": importance_type,
        }
    )


def validate_required_columns(data: pd.DataFrame) -> None:
    """Ensure the feature table contains all columns needed for training."""
    required_columns = MODEL_FEATURES + [TARGET_COLUMN, "region", "year", "month", "latitude", "longitude"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in features.csv: {missing_columns}. "
            "Run python src/features.py before training."
        )


def train_and_evaluate_split(
    validation_split: ValidationSplit,
    models: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    """Train every model on one validation split and return metrics and outputs."""
    x_train = validation_split.train_data[MODEL_FEATURES]
    y_train = validation_split.train_data[TARGET_COLUMN]
    x_test = validation_split.test_data[MODEL_FEATURES]
    y_test = validation_split.test_data[TARGET_COLUMN]

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    split_label = validation_split.strategy
    if validation_split.strategy == "region_transfer":
        split_label = f"{validation_split.train_region}_to_{validation_split.test_region}"

    for model_name, model in models.items():
        print(f"Training {model_name} on {split_label}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = calculate_metrics(y_test, y_pred)

        metrics_rows.append(
            {
                "validation_strategy": validation_split.strategy,
                "train_region": validation_split.train_region,
                "test_region": validation_split.test_region,
                "model": model_name,
                "n_train": len(validation_split.train_data),
                "n_test": len(validation_split.test_data),
                **metrics,
            }
        )

        predictions = validation_split.test_data[
            ["region", "year", "month", "latitude", "longitude"]
        ].copy()
        predictions["validation_strategy"] = validation_split.strategy
        predictions["train_region"] = validation_split.train_region
        predictions["test_region"] = validation_split.test_region
        predictions["model"] = model_name
        predictions["actual"] = y_test.to_numpy()
        predictions["prediction"] = y_pred
        prediction_frames.append(predictions)

        importance_frames.append(collect_feature_importance(model_name, model, validation_split))

    return metrics_rows, prediction_frames, importance_frames


def nested_metrics(metrics: pd.DataFrame, strategy: str) -> dict[str, Any]:
    """Convert one strategy's metrics rows into readable nested JSON."""
    strategy_metrics = metrics[metrics["validation_strategy"] == strategy]
    output: dict[str, Any] = {}

    for _, row in strategy_metrics.iterrows():
        if strategy == "region_transfer":
            key = f"{row['train_region']}_to_{row['test_region']}"
            output.setdefault(key, {})
            output[key][row["model"]] = {
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "R2": row["R2"],
                "n_train": int(row["n_train"]),
                "n_test": int(row["n_test"]),
            }
        else:
            output[row["model"]] = {
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "R2": row["R2"],
                "n_train": int(row["n_train"]),
                "n_test": int(row["n_test"]),
            }

    return output


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Save JSON with consistent formatting."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> None:
    """Run Phase 3 model comparison under all validation strategies."""
    ensure_directories()
    data = load_features()
    validate_required_columns(data)

    validation_splits = build_validation_splits(data)

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for validation_split in validation_splits:
        split_metrics, split_predictions, split_importance = train_and_evaluate_split(
            validation_split,
            build_models(),
        )
        metrics_rows.extend(split_metrics)
        prediction_frames.extend(split_predictions)
        importance_frames.extend(split_importance)

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    feature_importance = pd.concat(importance_frames, ignore_index=True)

    metrics.to_csv(ALL_METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    save_json(RANDOM_SPLIT_METRICS_PATH, nested_metrics(metrics, "random_split"))
    save_json(SPATIAL_HOLDOUT_METRICS_PATH, nested_metrics(metrics, "spatial_holdout"))
    save_json(REGION_TRANSFER_METRICS_PATH, nested_metrics(metrics, "region_transfer"))

    print(f"Saved all metrics to {ALL_METRICS_PATH}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Saved feature importance to {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved random split metrics to {RANDOM_SPLIT_METRICS_PATH}")
    print(f"Saved spatial holdout metrics to {SPATIAL_HOLDOUT_METRICS_PATH}")
    print(f"Saved region transfer metrics to {REGION_TRANSFER_METRICS_PATH}")
    print("Best rows by validation strategy:")
    print(metrics.sort_values("RMSE").groupby("validation_strategy").head(1).to_string(index=False))


if __name__ == "__main__":
    main()
