"""Benchmark data service — reads from outputs/benchmark/ CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import BENCHMARK_DIR, BENCHMARK_SPLITS_DIR, REGIONS
from backend.data_loader import read_csv_cached, read_json_cached


def _benchmark_path(filename: str) -> Path:
    return BENCHMARK_DIR / filename


def get_benchmark_summary() -> dict[str, Any]:
    """Top-level benchmark metadata."""
    lb = read_csv_cached(_benchmark_path("leaderboard.csv"))
    all_r = read_csv_cached(_benchmark_path("all_results.csv"))
    diff = read_csv_cached(_benchmark_path("split_difficulty_analysis.csv"))
    registry_path = BENCHMARK_DIR / "experiment_registry.json"
    reg = read_json_cached(registry_path)

    experiments = reg.get("experiments", [])
    completed = sum(1 for e in experiments if e.get("status") == "completed")
    failed = sum(1 for e in experiments if e.get("status") == "failed")

    return {
        "benchmark_name": "EvapAnomaly-Forecast-v1",
        "total_experiments": len(experiments),
        "completed": completed,
        "failed": failed,
        "n_models": lb["model_name"].nunique() if not lb.empty else 0,
        "n_split_protocols": lb["split_protocol"].nunique() if not lb.empty else 0,
        "n_feature_sets": lb["feature_set"].nunique() if not lb.empty else 0,
        "best_rmse": float(lb["rmse"].min()) if not lb.empty and "rmse" in lb.columns else None,
        "best_model": str(lb.loc[lb["rmse"].idxmin(), "model_name"]) if not lb.empty and "rmse" in lb.columns else None,
        "regions": REGIONS,
        "split_difficulty": (
            diff.to_dict(orient="records") if not diff.empty else []
        ),
    }


def get_benchmark_task() -> dict[str, Any]:
    """Static task definition."""
    return {
        "task_name": "Next-Month Land Evaporation Anomaly Forecasting",
        "input_window": "X_{t-6 : t-1}",
        "target": "evaporation_anomaly at month t",
        "spatial_unit": "grid cell (0.1° × 0.1°)",
        "temporal_unit": "monthly",
        "data_source": "ERA5-Land reanalysis",
        "forecast_horizon": "1 month",
        "sequence_length": 6,
    }


def get_benchmark_regions() -> list[dict[str, Any]]:
    """Return benchmark region definitions."""
    try:
        from climatenet.benchmark.region_registry import get_default_registry
        reg = get_default_registry()
        return [
            {
                "name": r.name,
                "lat_min": r.lat_min,
                "lat_max": r.lat_max,
                "lon_min": r.lon_min,
                "lon_max": r.lon_max,
                "climate_type": r.climate_type,
                "description": r.description,
            }
            for r in reg.list_all()
        ]
    except Exception:
        return []


def get_benchmark_splits() -> list[dict[str, Any]]:
    """Return available split protocols and their configs."""
    splits = []
    if BENCHMARK_SPLITS_DIR.exists():
        for d in sorted(BENCHMARK_SPLITS_DIR.iterdir()):
            if d.is_dir():
                meta = read_json_cached(d / "split_metadata.json")
                if meta:
                    splits.append(meta)
    return splits
