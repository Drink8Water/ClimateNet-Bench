"""Data loading functions for ClimateNet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from climatenet.utils.paths import resolve_project_path


def load_csv(path: str | Path, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV file and optionally validate required columns."""
    resolved_path = resolve_project_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_path}")

    data = pd.read_csv(resolved_path)
    if required_columns:
        missing = [column for column in required_columns if column not in data.columns]
        if missing:
            raise ValueError(f"{resolved_path} is missing required columns: {missing}")
    return data


def save_csv(data: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe to CSV."""
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
