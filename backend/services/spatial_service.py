"""Spatial grid and timeseries data access from features.csv."""

from __future__ import annotations

import pandas as pd

from backend.config import (
    ANOMALY_VARIABLES,
    FEATURES_PATH,
    REGIONS,
    get_experiment_dir,
)
from backend.data_loader import read_csv_cached


def get_spatial_grid(
    variable: str = "evaporation_anomaly",
    region: str | None = None,
    year: int | None = None,
    month: int | None = None,
    limit: int = 2000,
) -> list[dict]:
    """Return grid points with the selected anomaly variable value."""
    # Try experiment features first, fall back to processed features
    latest_dir = get_experiment_dir("latest")
    path = latest_dir / "features.csv"
    if not path.exists():
        path = FEATURES_PATH

    df = read_csv_cached(path)
    if df.empty:
        return []

    if region:
        df = df[df["region"] == region]
    if year:
        df = df[df["year"] == year]
    if month:
        df = df[df["month"] == month]

    value_col = variable if variable in df.columns else "evaporation_anomaly"

    df = df.head(limit)
    results = []
    for _, row in df.iterrows():
        results.append({
            "region": row.get("region", ""),
            "latitude": float(row["latitude"]) if pd.notna(row.get("latitude")) else 0.0,
            "longitude": float(row["longitude"]) if pd.notna(row.get("longitude")) else 0.0,
            "year": int(row["year"]) if pd.notna(row.get("year")) else 0,
            "month": int(row["month"]) if pd.notna(row.get("month")) else 0,
            "value": float(row[value_col]) if pd.notna(row.get(value_col)) else None,
        })
    return results


def get_timeseries(
    region: str | None = None,
    variable: str = "evaporation_anomaly",
) -> list[dict]:
    """Return monthly aggregated timeseries by region."""
    path = FEATURES_PATH
    df = read_csv_cached(path)
    if df.empty:
        return []

    if region:
        df = df[df["region"] == region]

    value_col = variable if variable in df.columns else "evaporation_anomaly"

    # Aggregate: mean per year-month-region
    group_cols = ["region", "year", "month"]
    agg_cols = {
        "temperature": "mean",
        "precipitation": "mean",
        "radiation": "mean",
        "soil_moisture": "mean",
        value_col: "mean",
    }
    available_agg = {k: v for k, v in agg_cols.items() if k in df.columns}
    grouped = df.groupby(group_cols, as_index=False).agg(available_agg)

    results = []
    for _, row in grouped.iterrows():
        record = {
            "region": row.get("region", ""),
            "year": int(row["year"]) if pd.notna(row.get("year")) else 0,
            "month": int(row["month"]) if pd.notna(row.get("month")) else 0,
            "temperature": float(row["temperature"]) if pd.notna(row.get("temperature")) else None,
            "precipitation": float(row["precipitation"]) if pd.notna(row.get("precipitation")) else None,
            "radiation": float(row["radiation"]) if pd.notna(row.get("radiation")) else None,
            "soil_moisture": float(row["soil_moisture"]) if pd.notna(row.get("soil_moisture")) else None,
            "evaporation_anomaly": float(row[value_col]) if pd.notna(row.get(value_col)) else None,
        }
        results.append(record)
    return results


def get_grid_cell_detail(
    latitude: float,
    longitude: float,
    year: int,
    month: int,
) -> dict | None:
    """Return full detail for a specific grid cell at a given time."""
    path = FEATURES_PATH
    df = read_csv_cached(path)
    if df.empty:
        return None

    # Find closest match (tolerance for floating point)
    mask = (
        (df["year"] == year)
        & (df["month"] == month)
        & (df["latitude"].round(4) == round(latitude, 4))
        & (df["longitude"].round(4) == round(longitude, 4))
    )
    matches = df[mask]
    if matches.empty:
        # Try without rounding
        mask = (
            (df["year"] == year)
            & (df["month"] == month)
            & (df["latitude"] == latitude)
            & (df["longitude"] == longitude)
        )
        matches = df[mask]
    if matches.empty:
        return None

    row = matches.iloc[0]
    return {
        "region": str(row.get("region", "")),
        "year": int(row["year"]),
        "month": int(row["month"]),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "temperature": float(row["temperature"]) if pd.notna(row.get("temperature")) else None,
        "precipitation": float(row["precipitation"]) if pd.notna(row.get("precipitation")) else None,
        "radiation": float(row["radiation"]) if pd.notna(row.get("radiation")) else None,
        "soil_moisture": float(row["soil_moisture"]) if pd.notna(row.get("soil_moisture")) else None,
        "evaporation": float(row["evaporation"]) if pd.notna(row.get("evaporation")) else None,
        "evaporation_anomaly": float(row["evaporation_anomaly"]) if pd.notna(row.get("evaporation_anomaly")) else None,
        "wind_speed": float(row["wind_speed"]) if pd.notna(row.get("wind_speed")) else None,
    }
