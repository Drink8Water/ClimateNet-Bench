"""Experiment-related data access — reads from local CSV/JSON files."""

from __future__ import annotations

import pandas as pd

from backend.config import (
    ALL_EXPERIMENTS_PATH,
    EXPERIMENTS_DIR,
    FEATURE_SETS,
    MODEL_NAMES,
    REGIONS,
    VALIDATION_STRATEGIES,
    get_experiment_dir,
    list_experiment_ids,
)
from backend.data_loader import read_csv_cached, read_json_cached, read_yaml_cached


def get_all_experiments(
    model_name: str | None = None,
    validation_strategy: str | None = None,
    feature_set: str | None = None,
    region: str | None = None,
) -> list[dict]:
    """Return experiments from all_experiments.csv with optional filters."""
    df = read_csv_cached(ALL_EXPERIMENTS_PATH)
    if df.empty:
        return []

    if model_name:
        df = df[df["model_name"] == model_name]
    if validation_strategy:
        df = df[df["validation_strategy"] == validation_strategy]
    if feature_set:
        df = df[df["feature_set"] == feature_set]
    if region:
        df = df[(df["train_region"].str.contains(region, na=False)) |
                (df["test_region"].str.contains(region, na=False))]

    # Normalize column names: uppercase metric columns -> lowercase for Pydantic
    rename_map = {}
    for col in ["MAE", "RMSE", "R2"]:
        if col in df.columns:
            rename_map[col] = col.lower()
    if rename_map:
        df = df.rename(columns=rename_map)

    # Fill NaN with None for JSON serialization
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def get_experiment_summary() -> dict:
    """Aggregated KPIs across all experiments."""
    df = read_csv_cached(ALL_EXPERIMENTS_PATH)
    if df.empty:
        return {
            "total_experiments": 0,
            "best_r2": None,
            "best_rmse": None,
            "best_model": None,
            "regions": REGIONS,
            "models": [],
            "strategies": [],
            "feature_sets": [],
        }

    best_r2_row = df.loc[df["R2"].idxmax()] if "R2" in df.columns and not df["R2"].isna().all() else None
    best_rmse_row = df.loc[df["RMSE"].idxmin()] if "RMSE" in df.columns and not df["RMSE"].isna().all() else None

    return {
        "total_experiments": len(df),
        "best_r2": float(best_r2_row["R2"]) if best_r2_row is not None else None,
        "best_rmse": float(best_rmse_row["RMSE"]) if best_rmse_row is not None else None,
        "best_model": str(best_r2_row["model_name"]) if best_r2_row is not None else None,
        "regions": REGIONS,
        "models": [m for m in MODEL_NAMES if m in df["model_name"].unique()],
        "strategies": [s for s in VALIDATION_STRATEGIES if s in df["validation_strategy"].unique()],
        "feature_sets": [f for f in FEATURE_SETS if f in df["feature_set"].unique()],
    }


def get_experiment_detail(experiment_id: str) -> dict | None:
    """Return config + metrics summary for a single experiment."""
    exp_dir = get_experiment_dir(experiment_id)
    if not exp_dir.exists():
        return None

    config = read_yaml_cached(exp_dir / "config.yaml")
    metrics = read_csv_cached(exp_dir / "metrics.csv")
    predictions_path = exp_dir / "predictions.csv"
    features_path = exp_dir / "features.csv"

    # Count rows efficiently
    pred_count = 0
    feat_count = 0
    if predictions_path.exists():
        try:
            pred_count = sum(1 for _ in open(predictions_path, encoding="utf-8")) - 1  # minus header
        except Exception:
            pass
    if features_path.exists():
        try:
            feat_count = sum(1 for _ in open(features_path, encoding="utf-8")) - 1
        except Exception:
            pass

    metrics_data = metrics.where(pd.notna(metrics), None).to_dict(orient="records") if not metrics.empty else []

    return {
        "experiment_id": experiment_id,
        "config": config,
        "metrics": metrics_data,
        "metrics_count": len(metrics_data),
        "prediction_count": max(pred_count, 0),
        "feature_count": max(feat_count, 0),
    }


def get_experiment_metrics(experiment_id: str) -> list[dict]:
    """Return metrics rows for one experiment."""
    exp_dir = get_experiment_dir(experiment_id)
    df = read_csv_cached(exp_dir / "metrics.csv")
    if df.empty:
        return []
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")
