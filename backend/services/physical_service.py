"""Physical consistency audit data service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import BENCHMARK_DIR
from backend.data_loader import read_csv_cached, read_json_cached


_PHYSICAL_DIR = BENCHMARK_DIR / "physical_consistency"


def get_physical_summary() -> dict[str, Any]:
    """Return the physical consistency audit summary."""
    summary_path = _PHYSICAL_DIR / "consistency_summary.json"
    data = read_json_cached(summary_path)
    if not data:
        return {"message": "No physical consistency audit data found."}
    return data


def get_regional_sensitivity(
    feature: str | None = None,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Return regional sensitivity table, optionally filtered."""
    path = _PHYSICAL_DIR / "regional_sensitivity.csv"
    df = read_csv_cached(path)
    if df.empty:
        return []
    if feature:
        df = df[df["feature"] == feature]
    if region:
        df = df[df["region"] == region]
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")
