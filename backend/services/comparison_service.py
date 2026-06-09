"""Model comparison and ablation study data access."""

from __future__ import annotations

import pandas as pd

from backend.config import get_experiment_dir
from backend.data_loader import read_csv_cached
from backend.services.experiment_service import get_all_experiments


def get_model_comparison(
    metric: str = "rmse",
    validation_strategy: str | None = None,
    feature_set: str | None = None,
) -> list[dict]:
    """Return metric values grouped by model, suitable for bar charts."""
    experiments = get_all_experiments(
        validation_strategy=validation_strategy,
        feature_set=feature_set,
    )
    if not experiments:
        return []

    metric_key = metric.upper()
    results = []
    for exp in experiments:
        value = exp.get(metric_key) or exp.get(metric.lower())
        results.append({
            "model_name": exp.get("model_name", ""),
            "validation_strategy": exp.get("validation_strategy", ""),
            "feature_set": exp.get("feature_set", ""),
            "train_region": exp.get("train_region", ""),
            "test_region": exp.get("test_region", ""),
            "mae": exp.get("MAE") or exp.get("mae"),
            "rmse": exp.get("RMSE") or exp.get("rmse"),
            "r2": exp.get("R2") or exp.get("r2"),
            "value": value,
        })
    return results


def get_ablation_study(experiment_id: str = "latest") -> list[dict]:
    """Return ablation study data (feature set comparison)."""
    exp_dir = get_experiment_dir(experiment_id)
    path = exp_dir / "ablation_study.csv"
    df = read_csv_cached(path)
    if df.empty:
        return []
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")
