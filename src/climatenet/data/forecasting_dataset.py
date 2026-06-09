"""Forecasting dataset constructor for ClimateNet-Bench.

Builds supervised learning samples from processed climate feature tables:

- **Flattened format** (for tree / linear models):
  Each sample is one row with columns ``temperature_anomaly_lag_1`` …
  ``temperature_anomaly_lag_6``, etc.

- **3D sequence format** (for TCN / deep models):
  Each sample is a ``(sequence_length, n_features)`` array.

Lag convention
--------------
- ``lag_1`` = the month immediately before the target month (t−1).
- ``lag_6`` = six months before the target month (t−6).
- The target month ``t`` is **never** included in the input window.

Anti-leakage guarantees
-----------------------
- Features for month ``t`` are built exclusively from months ``t−6 … t−1``.
- The target ``y = evaporation_anomaly`` at month ``t`` is never seen
  by the feature builder.
- Incomplete histories (grid cells with fewer than 7 consecutive months)
  are silently dropped — a warning is logged when this happens.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.benchmark.region_registry import RegionRegistry, get_default_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_COLUMNS: list[str] = [
    "temperature_anomaly",
    "precipitation_anomaly",
    "radiation_anomaly",
    "soil_moisture_anomaly",
    "wind_speed",
    "dryness_proxy",
    "saturation_vapor_pressure",
]

STATIC_FEATURE_COLUMNS: list[str] = [
    "month_sin",
    "month_cos",
    "latitude",
    "longitude",
]

# All feature columns that should be lagged or included as static context.
ALL_FEATURE_COLUMNS: list[str] = DEFAULT_FEATURE_COLUMNS + STATIC_FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


@dataclass
class ForecastingMetadata:
    """Metadata for a built forecasting dataset."""

    total_samples: int
    feature_columns: list[str]
    static_columns: list[str]
    sequence_length: int
    target_column: str
    input_window_start: str = ""  # e.g. "t-6"
    input_window_end: str = ""  # e.g. "t-1"
    lag_convention: str = "lag_1 = previous month (t-1), lag_6 = six months before target (t-6)"
    grid_cells: int = 0
    regions: list[str] = field(default_factory=list)
    climate_types: list[str] = field(default_factory=list)
    dropped_incomplete: int = 0
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Grid-cell helpers
# ---------------------------------------------------------------------------


def make_grid_id(latitude: float, longitude: float) -> str:
    """Create a stable grid-cell identifier from coordinates.

    >>> make_grid_id(20.1234, -15.5678)
    '20.1234_-15.5678'
    """
    return f"{latitude:.4f}_{longitude:.4f}"


def make_sample_id(
    region: str,
    grid_id: str,
    target_year: int,
    target_month: int,
) -> str:
    """Create a unique, human-readable sample identifier.

    >>> make_sample_id("Sahara", "20.0_10.0", 2020, 7)
    'Sahara_20.0_10.0_2020_07'
    """
    return f"{region}_{grid_id}_{target_year}_{target_month:02d}"


# ---------------------------------------------------------------------------
# Core window builder
# ---------------------------------------------------------------------------


def build_forecasting_samples(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    static_columns: list[str] | None = None,
    target_column: str = "evaporation_anomaly",
    sequence_length: int = 6,
    registry: RegionRegistry | None = None,
) -> tuple[pd.DataFrame, ForecastingMetadata]:
    """Build a flattened forecasting sample table from a processed feature table.

    Parameters
    ----------
    data
        Processed feature table. Must include columns: ``region``, ``year``,
        ``month``, ``latitude``, ``longitude``, all ``feature_columns``,
        all ``static_columns``, and ``target_column``.
    feature_columns
        Columns that vary over time and should be lagged.  Defaults to
        ``DEFAULT_FEATURE_COLUMNS``.
    static_columns
        Columns treated as static context (not lagged).  Defaults to
        ``STATIC_FEATURE_COLUMNS``.  ``month_sin`` and ``month_cos`` are
        taken from the **target** month so they encode "which month are we
        predicting for?"
    target_column
        Name of the target column (default ``"evaporation_anomaly"``).
    sequence_length
        Number of past months to include (default 6).
    registry
        Optional :class:`RegionRegistry` for ``climate_type`` lookup.
        Uses the default registry if not provided.

    Returns
    -------
    (samples_df, metadata)
        ``samples_df`` has one row per forecasting sample with lagged
        feature columns, static columns, identifiers, and ``y_true``.
    """
    # ------------------------------------------------------------------
    # defaults
    # ------------------------------------------------------------------
    if feature_columns is None:
        feature_columns = list(DEFAULT_FEATURE_COLUMNS)
    if static_columns is None:
        static_columns = list(STATIC_FEATURE_COLUMNS)
    if registry is None:
        registry = get_default_registry()

    # ------------------------------------------------------------------
    # validate required columns
    # ------------------------------------------------------------------
    required = [
        "region",
        "year",
        "month",
        "latitude",
        "longitude",
        *feature_columns,
        target_column,
    ]
    # static columns: latitude/longitude are already in required;
    # month_sin/month_cos should be taken from the target row
    for col in static_columns:
        if col not in ("latitude", "longitude") and col not in required:
            # Some static columns (like month_sin, month_cos) vary per row
            # but we treat them as point-in-time context, not lagged.
            pass
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in feature table: {missing}")

    # ------------------------------------------------------------------
    # sorted time series per grid cell
    # ------------------------------------------------------------------
    group_keys = ["region", "latitude", "longitude"]
    rows: list[dict[str, Any]] = []
    total_dropped = 0
    grid_cell_count = 0
    # Also lag the target column so persistence baseline can use lag_1 target.
    lag_target = True

    for (region, lat, lon), group in data.groupby(group_keys):
        grid_cell_count += 1
        grid_id = make_grid_id(lat, lon)

        # Chronological sort — required for correct lag construction.
        group = group.sort_values(["year", "month"]).reset_index(drop=True)

        # A grid cell needs at least sequence_length + 1 rows to produce
        # one sample (6 input months + 1 target month).
        min_rows = sequence_length + 1
        if len(group) < min_rows:
            total_dropped += len(group)
            continue

        n_months = len(group)
        for t in range(sequence_length, n_months):
            # t = index of the target month.
            # Input window = t - sequence_length ... t - 1.
            window_start = t - sequence_length
            window_end = t - 1  # inclusive

            target_row = group.iloc[t]
            target_year = int(target_row["year"])
            target_month = int(target_row["month"])

            # --- build sample row ---
            sample: dict[str, Any] = {}

            # identifiers
            sample["sample_id"] = make_sample_id(
                str(region), grid_id, target_year, target_month
            )
            sample["grid_id"] = grid_id
            sample["region"] = str(region)
            sample["target_year"] = target_year
            sample["target_month"] = target_month
            sample["latitude"] = float(lat)
            sample["longitude"] = float(lon)

            # climate_type from registry
            try:
                r = registry.get(str(region))
                sample["climate_type"] = r.climate_type
            except KeyError:
                sample["climate_type"] = "unknown"

            # input window boundaries
            input_start_row = group.iloc[window_start]
            input_end_row = group.iloc[window_end]
            sample["input_window_start"] = (
                f"{int(input_start_row['year'])}-{int(input_start_row['month']):02d}"
            )
            sample["input_window_end"] = (
                f"{int(input_end_row['year'])}-{int(input_end_row['month']):02d}"
            )

            # --- lagged features ---
            # lag_1 = t-1 (previous month), lag_6 = t-6 (six months before target)
            for lag_idx in range(1, sequence_length + 1):
                source_idx = t - lag_idx  # t-1, t-2, ..., t-6
                source_row = group.iloc[source_idx]
                for feat in feature_columns:
                    col_name = f"{feat}_lag_{lag_idx}"
                    sample[col_name] = float(source_row[feat])

            # --- static context from target month ---
            # month_sin / month_cos encode *which* month we predict for.
            # latitude / longitude are already set above.
            for feat in static_columns:
                if feat in ("latitude", "longitude"):
                    continue  # already set as identifiers
                if feat in target_row.index:
                    sample[feat] = float(target_row[feat])

            # --- lagged target (for persistence baseline only) ---
            # persistence model uses y_{t-1} as its prediction.
            # This column is metadata, NOT a model feature.
            if lag_target:
                source_idx = t - 1  # previous month
                source_row = group.iloc[source_idx]
                sample[f"{target_column}_lag_1"] = float(source_row[target_column])

            # --- target ---
            sample["y_true"] = float(target_row[target_column])

            rows.append(sample)

    if not rows:
        raise ValueError(
            "No forecasting samples were generated. "
            f"Check that grid cells have at least {sequence_length + 1} "
            "consecutive monthly records."
        )

    samples_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    input_start = samples_df["input_window_start"].iloc[0] if len(samples_df) > 0 else ""
    input_end = samples_df["input_window_end"].iloc[0] if len(samples_df) > 0 else ""

    metadata = ForecastingMetadata(
        total_samples=len(samples_df),
        feature_columns=feature_columns,
        static_columns=static_columns,
        sequence_length=sequence_length,
        target_column=target_column,
        input_window_start=input_start,
        input_window_end=input_end,
        grid_cells=grid_cell_count,
        regions=sorted(samples_df["region"].unique().tolist()),
        climate_types=sorted(samples_df["climate_type"].unique().tolist()),
        dropped_incomplete=total_dropped,
    )

    if total_dropped > 0:
        logger.info(
            "Dropped %d rows from grid cells with fewer than %d months of history.",
            total_dropped,
            sequence_length + 1,
        )

    return samples_df, metadata


# ---------------------------------------------------------------------------
# 3D sequence format (for TCN / deep models)
# ---------------------------------------------------------------------------


def build_sequence_arrays_from_samples(
    samples_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    sequence_length: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a flattened forecasting sample table into 3D sequence arrays.

    Returns ``(X, y)`` where ``X`` has shape ``(n_samples, sequence_length,
    n_features)`` and ``y`` has shape ``(n_samples,)``.

    This is the inverse of flattening — it reconstructs the temporal
    dimension from the lagged columns.
    """
    if feature_columns is None:
        feature_columns = list(DEFAULT_FEATURE_COLUMNS)

    n_samples = len(samples_df)
    n_features = len(feature_columns)
    X = np.zeros((n_samples, sequence_length, n_features), dtype=np.float32)

    for lag_idx in range(1, sequence_length + 1):
        array_idx = lag_idx - 1  # 0-indexed array position
        for feat_idx, feat in enumerate(feature_columns):
            col = f"{feat}_lag_{lag_idx}"
            if col in samples_df.columns:
                X[:, array_idx, feat_idx] = samples_df[col].to_numpy(dtype=np.float32)

    y = samples_df["y_true"].to_numpy(dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_samples(samples_df: pd.DataFrame, sequence_length: int = 6) -> list[str]:
    """Run all validation checks on a forecasting sample table.

    Returns a list of error messages (empty list = all checks pass).

    Checks performed:

    1. No duplicated ``sample_id`` values.
    2. All expected lag columns present.
    3. No future leakage (input_window_end < target date).
    4. Chronological order within each grid cell.
    """
    errors: list[str] = []

    errors.extend(_check_no_duplicate_ids(samples_df))
    errors.extend(_check_lag_columns(samples_df, sequence_length))
    errors.extend(_check_no_future_leakage(samples_df))
    errors.extend(_check_chronological_order(samples_df))

    return errors


def _check_no_duplicate_ids(samples_df: pd.DataFrame) -> list[str]:
    """Verify that sample_id is unique."""
    dupes = samples_df["sample_id"].duplicated()
    if dupes.any():
        dup_ids = samples_df.loc[dupes, "sample_id"].unique().tolist()
        return [f"Duplicate sample_id values found: {dup_ids}"]
    return []


def _check_lag_columns(samples_df: pd.DataFrame, sequence_length: int) -> list[str]:
    """Verify all expected lag columns exist."""
    errors: list[str] = []
    for feat in DEFAULT_FEATURE_COLUMNS:
        for lag in range(1, sequence_length + 1):
            col = f"{feat}_lag_{lag}"
            if col not in samples_df.columns:
                errors.append(f"Missing lag column: {col}")
    return errors


def _check_no_future_leakage(samples_df: pd.DataFrame) -> list[str]:
    """Verify input_window_end is strictly before the target month."""
    errors: list[str] = []
    for _, row in samples_df.iterrows():
        target_str = f"{int(row['target_year'])}-{int(row['target_month']):02d}"
        input_end = str(row["input_window_end"])

        # Simple string comparison works for YYYY-MM format
        if input_end >= target_str:
            errors.append(
                f"Future leakage in sample {row['sample_id']}: "
                f"input_window_end={input_end} >= target={target_str}"
            )
        if len(errors) >= 10:
            errors.append("... (truncated after 10 leakage errors)")
            break
    return errors


def _check_chronological_order(samples_df: pd.DataFrame) -> list[str]:
    """Verify that within each grid_id, samples are in chronological order."""
    errors: list[str] = []
    for grid_id, group in samples_df.groupby("grid_id"):
        sorted_group = group.sort_values(["target_year", "target_month"])
        for i in range(1, len(sorted_group)):
            prev = sorted_group.iloc[i - 1]
            curr = sorted_group.iloc[i]
            prev_date = (int(prev["target_year"]), int(prev["target_month"]))
            curr_date = (int(curr["target_year"]), int(curr["target_month"]))
            if curr_date <= prev_date:
                errors.append(
                    f"Non-chronological order in grid {grid_id}: "
                    f"{prev_date} -> {curr_date}"
                )
                if len(errors) >= 10:
                    break
        if len(errors) >= 10:
            break
    return errors


# ---------------------------------------------------------------------------
# Dataset-level I/O
# ---------------------------------------------------------------------------


def save_forecasting_dataset(
    samples_df: pd.DataFrame,
    metadata: ForecastingMetadata,
    output_dir: str | Path,
    prefix: str = "forecasting",
) -> dict[str, Path]:
    """Save samples and metadata to disk.

    Returns a dict mapping artifact names to saved paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples_path = out / f"{prefix}_samples.csv"
    metadata_path = out / f"{prefix}_metadata.json"

    samples_df.to_csv(samples_path, index=False)

    meta_dict: dict[str, Any] = {
        "total_samples": metadata.total_samples,
        "feature_columns": metadata.feature_columns,
        "static_columns": metadata.static_columns,
        "sequence_length": metadata.sequence_length,
        "target_column": metadata.target_column,
        "lag_convention": metadata.lag_convention,
        "grid_cells": metadata.grid_cells,
        "regions": metadata.regions,
        "climate_types": metadata.climate_types,
        "dropped_incomplete": metadata.dropped_incomplete,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, default=str)

    return {"samples": samples_path, "metadata": metadata_path}


def load_forecasting_dataset(
    samples_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load a previously saved forecasting dataset.

    Returns ``(samples_df, metadata_dict)``.  ``metadata_dict`` is ``None``
    when ``metadata_path`` is not provided.
    """
    df = pd.read_csv(samples_path)
    meta = None
    if metadata_path is not None:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return df, meta


# ---------------------------------------------------------------------------
# Convenience: build + validate + save
# ---------------------------------------------------------------------------


def build_and_save(
    data: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "forecasting",
    feature_columns: list[str] | None = None,
    target_column: str = "evaporation_anomaly",
    sequence_length: int = 6,
    registry: RegionRegistry | None = None,
) -> tuple[pd.DataFrame, ForecastingMetadata, list[str]]:
    """Run the full pipeline: build, validate, and save.

    Returns ``(samples_df, metadata, validation_errors)``.

    Raises ``ValueError`` if validation errors are found.
    """
    samples_df, metadata = build_forecasting_samples(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
        registry=registry,
    )

    errors = validate_samples(samples_df, sequence_length=sequence_length)

    save_forecasting_dataset(samples_df, metadata, output_dir, prefix=prefix)

    if errors:
        logger.warning("Validation warnings for forecasting dataset:\n%s", "\n".join(errors))

    return samples_df, metadata, errors
