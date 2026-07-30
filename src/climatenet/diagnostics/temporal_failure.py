"""Bounded diagnostics for ERA5-Land temporal generalisation failures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


SEASON_BY_MONTH = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def compute_error_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Compute signed and tail-aware prediction diagnostics."""
    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
    residual = y_pred - y_true
    absolute = np.abs(residual)
    if len(frame) == 0:
        raise ValueError("Cannot compute metrics for an empty frame")
    return {
        "n": int(len(frame)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(np.mean(residual)),
        "r2": float(r2_score(y_true, y_pred)),
        "residual_p50": float(np.quantile(residual, 0.50)),
        "residual_p90": float(np.quantile(residual, 0.90)),
        "residual_p95": float(np.quantile(residual, 0.95)),
        "residual_p99": float(np.quantile(residual, 0.99)),
        "abs_residual_p50": float(np.quantile(absolute, 0.50)),
        "abs_residual_p90": float(np.quantile(absolute, 0.90)),
        "abs_residual_p95": float(np.quantile(absolute, 0.95)),
        "abs_residual_p99": float(np.quantile(absolute, 0.99)),
        "overprediction_ratio": float(np.mean(residual > 0)),
        "underprediction_ratio": float(np.mean(residual < 0)),
        "y_true_mean": float(np.mean(y_true)),
        "y_true_std": float(np.std(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_pred_std": float(np.std(y_pred)),
        "prediction_to_target_std_ratio": (
            float(np.std(y_pred) / np.std(y_true))
            if np.std(y_true) > 0
            else float("nan")
        ),
    }


def aggregate_prediction_metrics(
    frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else group_columns
    )
    for keys, group in frame.groupby(grouper, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                **compute_error_metrics(group),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    samples: list[np.ndarray] = field(default_factory=list)

    def update(self, values: np.ndarray, sample_stride: int = 20) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.count += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        self.total_sq += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.samples.append(values[::sample_stride].copy())

    def summary(self) -> dict[str, float | int]:
        if not self.count:
            raise ValueError("No observations accumulated")
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        sample = np.concatenate(self.samples)
        return {
            "count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": self.minimum,
            "p10": float(np.quantile(sample, 0.10)),
            "p50": float(np.quantile(sample, 0.50)),
            "p90": float(np.quantile(sample, 0.90)),
            "max": self.maximum,
            "quantile_sample_count": int(len(sample)),
        }


def summarize_feature_shift(
    train_values: Iterable[float],
    test_values: Iterable[float],
) -> dict[str, float | int]:
    """Summarize one train/test feature pair; used by tests and CLI."""
    train = np.asarray(list(train_values), dtype=np.float64)
    test = np.asarray(list(test_values), dtype=np.float64)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if train.size == 0 or test.size == 0:
        raise ValueError("Feature shift requires non-empty finite train/test")
    train_std = float(np.std(train))
    test_std = float(np.std(test))
    pooled = float(np.sqrt((train_std**2 + test_std**2) / 2))
    return {
        "train_count": int(train.size),
        "test_count": int(test.size),
        "train_mean": float(np.mean(train)),
        "test_mean": float(np.mean(test)),
        "train_std": train_std,
        "test_std": test_std,
        "standardized_mean_difference": (
            float((np.mean(test) - np.mean(train)) / pooled)
            if pooled > 0
            else 0.0
        ),
        "train_p10": float(np.quantile(train, 0.10)),
        "test_p10": float(np.quantile(test, 0.10)),
        "train_p50": float(np.quantile(train, 0.50)),
        "test_p50": float(np.quantile(test, 0.50)),
        "train_p90": float(np.quantile(train, 0.90)),
        "test_p90": float(np.quantile(test, 0.90)),
        "test_outside_train_range_ratio": float(
            np.mean((test < np.min(train)) | (test > np.max(train)))
        ),
    }


def shift_row(
    feature: str,
    train: dict[str, float | int],
    test: dict[str, float | int],
    *,
    outside_ratio: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pooled = float(
        np.sqrt((float(train["std"]) ** 2 + float(test["std"]) ** 2) / 2)
    )
    return {
        "feature": feature,
        **(extra or {}),
        **{f"train_{key}": value for key, value in train.items()},
        **{f"test_{key}": value for key, value in test.items()},
        "standardized_mean_difference": (
            (float(test["mean"]) - float(train["mean"])) / pooled
            if pooled > 0
            else 0.0
        ),
        "p10_shift": float(test["p10"]) - float(train["p10"]),
        "p50_shift": float(test["p50"]) - float(train["p50"]),
        "p90_shift": float(test["p90"]) - float(train["p90"]),
        "test_outside_train_range_ratio": outside_ratio,
    }


def validate_repeated_spatial_plan(plan: dict[str, Any]) -> None:
    folds = int(plan.get("folds", 0))
    if folds < 3 or folds > 5:
        raise ValueError("Repeated spatial plan requires 3-5 folds")
    if not plan.get("region_stratified"):
        raise ValueError("Spatial folds must be stratified within each region")
    if not plan.get("no_grid_cell_overlap"):
        raise ValueError("Spatial folds must prohibit grid-cell overlap")
    if set(plan.get("models", [])) - {"linear_regression", "lightgbm"}:
        raise ValueError("Repeated spatial plan contains an unsupported model")
    if plan.get("feature_sets") != ["full"]:
        raise ValueError("Repeated spatial plan must use only full features")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
