"""Leaderboard data service."""

from __future__ import annotations

from typing import Any

import math

import pandas as pd

from backend.config import BENCHMARK_DIR
from backend.data_loader import read_csv_cached


def _sanitize(records: list[dict]) -> list[dict]:
    """Replace NaN with None for JSON serialisation."""
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records


def _path(name: str) -> Path:
    return BENCHMARK_DIR / name


def get_leaderboard(
    model_name: str | None = None,
    split_protocol: str | None = None,
    feature_set: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return ranked leaderboard with optional filters."""
    df = read_csv_cached(_path("leaderboard.csv"))
    if df.empty:
        return []

    if model_name:
        df = df[df["model_name"] == model_name]
    if split_protocol:
        df = df[df["split_protocol"] == split_protocol]
    if feature_set:
        df = df[df["feature_set"] == feature_set]

    df = df.head(limit)
    df = df.where(pd.notna(df), None)
    return _sanitize(df.to_dict(orient="records"))


def get_split_difficulty() -> list[dict[str, Any]]:
    """Return split difficulty analysis table."""
    df = read_csv_cached(_path("split_difficulty_analysis.csv"))
    if df.empty:
        return []
    df = df.where(pd.notna(df), None)
    return _sanitize(df.to_dict(orient="records"))


def get_ablation_study(
    model_name: str | None = None,
    split_protocol: str | None = None,
) -> list[dict[str, Any]]:
    """Return ablation study results."""
    df = read_csv_cached(_path("ablation_results.csv"))
    if df.empty:
        return []
    if model_name:
        df = df[df["model_name"] == model_name]
    if split_protocol:
        df = df[df["split_protocol"] == split_protocol]
    df = df.where(pd.notna(df), None)
    return _sanitize(df.to_dict(orient="records"))
