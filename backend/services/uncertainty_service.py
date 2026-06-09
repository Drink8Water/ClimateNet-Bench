"""Uncertainty / calibration data service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import BENCHMARK_DIR, BENCHMARK_EXPERIMENTS_DIR
from backend.data_loader import read_csv_cached


def get_calibration(
    model_name: str | None = None,
    split_protocol: str | None = None,
) -> list[dict[str, Any]]:
    """Return uncertainty calibration table."""
    df = read_csv_cached(BENCHMARK_DIR / "uncertainty_calibration.csv")
    if df.empty:
        return []
    if model_name:
        df = df[df["model_name"] == model_name]
    if split_protocol:
        df = df[df["split_protocol"] == split_protocol]
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def get_experiment_intervals(
    experiment_id: str,
    limit: int = 500,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Return prediction intervals from one experiment."""
    path = BENCHMARK_EXPERIMENTS_DIR / experiment_id / "intervals.csv"
    df = read_csv_cached(path)
    if df.empty:
        return []
    if region and "region" in df.columns:
        df = df[df["region"] == region]
    df = df.head(limit)
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")
