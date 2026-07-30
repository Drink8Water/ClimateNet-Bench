"""Audit-only repeated, region-stratified spatial fold generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class RepeatedSpatialDesign:
    fold_count: int = 5
    block_size_deg: float = 5.0
    random_seed: int = 42
    assignment_method: str = "geographic_interleaving"


def add_spatial_block_id(
    frame: pd.DataFrame,
    block_size_deg: float,
    *,
    latitude: str = "latitude",
    longitude: str = "longitude",
) -> pd.DataFrame:
    if block_size_deg <= 0:
        raise ValueError("block_size_deg must be positive")
    result = frame.copy()
    lat_block = (
        np.floor(result[latitude].to_numpy() / block_size_deg)
        * block_size_deg
    ).astype(int)
    lon_block = (
        np.floor(result[longitude].to_numpy() / block_size_deg)
        * block_size_deg
    ).astype(int)
    result["spatial_block_id"] = pd.Categorical(
        [
            f"block_lat{lat}_lon{lon}"
            for lat, lon in zip(lat_block, lon_block)
        ]
    )
    return result


def unique_grid_table(
    frame: pd.DataFrame,
    block_size_deg: float,
    *,
    region: str = "region",
    latitude: str = "latitude",
    longitude: str = "longitude",
) -> pd.DataFrame:
    grids = frame[[region, latitude, longitude]].drop_duplicates()
    grids = add_spatial_block_id(
        grids,
        block_size_deg,
        latitude=latitude,
        longitude=longitude,
    )
    return grids.rename(columns={region: "region"})


def _balanced_bins(
    block_counts: pd.DataFrame,
    fold_count: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    if len(block_counts) < fold_count:
        raise ValueError(
            f"Region has {len(block_counts)} blocks but needs {fold_count}"
        )
    ordered = block_counts.copy()
    ordered["_tie"] = rng.random(len(ordered))
    ordered = ordered.sort_values(
        ["grid_count", "_tie"], ascending=[False, True]
    )
    totals = np.zeros(fold_count, dtype=np.int64)
    bins: dict[str, int] = {}
    initial_order = rng.permutation(fold_count)
    for position, row in enumerate(ordered.itertuples(index=False)):
        if position < fold_count:
            fold = int(initial_order[position])
        else:
            candidates = np.flatnonzero(totals == totals.min())
            fold = int(rng.choice(candidates))
        block_id = str(row.spatial_block_id)
        bins[block_id] = fold
        totals[fold] += int(row.grid_count)
    return bins


def _geographic_interleaved_bins(
    block_counts: pd.DataFrame,
    fold_count: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Interleave a lat/lon snake traversal without using target values."""
    if len(block_counts) < fold_count:
        raise ValueError(
            f"Region has {len(block_counts)} blocks but needs {fold_count}"
        )
    parsed = block_counts.copy()
    coordinates = parsed["spatial_block_id"].str.extract(
        r"^block_lat(-?\d+)_lon(-?\d+)$"
    )
    if coordinates.isna().any().any():
        raise ValueError("Cannot parse spatial block coordinates")
    parsed["_lat_block"] = coordinates[0].astype(int)
    parsed["_lon_block"] = coordinates[1].astype(int)
    sequence: list[str] = []
    for band_position, (_, band) in enumerate(
        parsed.groupby("_lat_block", sort=True)
    ):
        ordered = band.sort_values(
            "_lon_block", ascending=(band_position % 2 == 0)
        )
        sequence.extend(ordered["spatial_block_id"].astype(str).tolist())
    offset = int(rng.integers(0, fold_count))
    return {
        block_id: int((position + offset) % fold_count)
        for position, block_id in enumerate(sequence)
    }


def generate_block_test_folds(
    grids: pd.DataFrame,
    design: RepeatedSpatialDesign,
) -> pd.DataFrame:
    """Assign each region/block to exactly one balanced test fold."""
    required = {"region", "spatial_block_id"}
    if not required.issubset(grids):
        raise ValueError(f"Grid table missing columns: {sorted(required-grids.columns)}")
    block_counts = (
        grids.groupby(["region", "spatial_block_id"], observed=True)
        .size()
        .rename("grid_count")
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for region_position, (region, regional) in enumerate(
        block_counts.groupby("region", observed=True, sort=True)
    ):
        rng = np.random.default_rng(design.random_seed + region_position)
        if design.assignment_method == "geographic_interleaving":
            bins = _geographic_interleaved_bins(
                regional, design.fold_count, rng
            )
        elif design.assignment_method == "balanced_grid_count":
            bins = _balanced_bins(regional, design.fold_count, rng)
        else:
            raise ValueError(
                f"Unknown assignment_method: {design.assignment_method}"
            )
        for row in regional.itertuples(index=False):
            rows.append(
                {
                    "region": str(region),
                    "spatial_block_id": str(row.spatial_block_id),
                    "grid_count": int(row.grid_count),
                    "test_fold": int(bins[str(row.spatial_block_id)]),
                }
            )
    assignments = pd.DataFrame(rows)
    if assignments.duplicated(["region", "spatial_block_id"]).any():
        raise ValueError("Duplicate region/block assignment generated")
    return assignments


def fold_block_assignments(
    block_test_folds: pd.DataFrame,
    fold: int,
    fold_count: int,
) -> pd.DataFrame:
    """Materialize train/validation/test block roles for one fold."""
    if fold < 0 or fold >= fold_count:
        raise ValueError(f"fold must be in [0, {fold_count})")
    result = block_test_folds.copy()
    validation_test_fold = (fold + 1) % fold_count
    result["partition"] = np.select(
        [
            result["test_fold"].eq(fold),
            result["test_fold"].eq(validation_test_fold),
        ],
        ["test", "validation"],
        default="train",
    )
    return result


def assign_row_partitions(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
) -> pd.Series:
    partition = pd.Series(index=frame.index, dtype="string")
    for region, regional in fold_assignments.groupby(
        "region", observed=True
    ):
        mapping = regional.set_index("spatial_block_id")[
            "partition"
        ].to_dict()
        mask = frame["region"].astype("string").eq(str(region))
        partition.loc[mask] = (
            frame.loc[mask, "spatial_block_id"]
            .astype("string")
            .map(mapping)
        )
    if partition.isna().any():
        raise ValueError("Some rows do not have repeated spatial assignments")
    return partition


def validate_fold_isolation(
    grids: pd.DataFrame,
    fold_assignments: pd.DataFrame,
) -> dict[str, int]:
    assigned = grids.merge(
        fold_assignments[
            ["region", "spatial_block_id", "partition"]
        ],
        on=["region", "spatial_block_id"],
        how="left",
        validate="many_to_one",
    )
    duplicate_assignment = int(
        fold_assignments.duplicated(
            ["region", "spatial_block_id"], keep=False
        ).sum()
    )
    grid_partition_counts = assigned.groupby(
        ["region", "latitude", "longitude"], observed=True
    )["partition"].nunique()
    leakage = int((grid_partition_counts > 1).sum())
    unassigned = int(assigned["partition"].isna().sum())
    return {
        "grid_leakage_count": leakage,
        "duplicate_assignment_count": duplicate_assignment,
        "unassigned_grid_count": unassigned,
    }


def fit_train_only_target_anomaly(
    frame: pd.DataFrame,
    partition: pd.Series,
    *,
    value_column: str = "evaporation",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit region/month evaporation climatology on train and transform all."""
    train = frame.loc[
        partition.eq("train"), ["region", "month", value_column]
    ]
    climatology = (
        train.groupby(["region", "month"], observed=True)[value_column]
        .mean()
        .rename("train_climatology")
        .reset_index()
    )
    regions = frame["region"].astype("string").to_numpy()
    months = frame["month"].to_numpy()
    values = frame[value_column].to_numpy(dtype=np.float64)
    anomaly = np.empty(len(frame), dtype=np.float64)
    filled = np.zeros(len(frame), dtype=bool)
    for row in climatology.itertuples(index=False):
        mask = (regions == str(row.region)) & (months == int(row.month))
        anomaly[mask] = values[mask] - float(row.train_climatology)
        filled[mask] = True
    if not filled.all():
        raise ValueError("Train climatology cannot transform every region/month")
    return anomaly, climatology


def array_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Cannot summarize an empty array")
    return {
        "sample_count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "max": float(finite.max()),
        "zero_baseline_rmse": float(np.sqrt(np.mean(np.square(finite)))),
    }


def evaluate_fold_acceptance(
    balance_rows: pd.DataFrame,
    *,
    acceptance: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate per-fold and across-fold acceptance criteria."""
    result = balance_rows.copy()
    median_std = float(result["test_target_std"].median())
    zero_mean = float(result["zero_baseline_rmse"].mean())
    zero_cv = (
        float(result["zero_baseline_rmse"].std(ddof=0) / zero_mean)
        if zero_mean
        else float("inf")
    )
    checks = {
        "test_east_china_share": result["test_east_china_share"].between(
            float(acceptance["min_test_east_china_share"]),
            float(acceptance["max_test_east_china_share"]),
        ),
        "validation_east_china_share": result[
            "validation_east_china_share"
        ].between(
            float(acceptance["min_validation_east_china_share"]),
            float(acceptance["max_validation_east_china_share"]),
        ),
        "test_regions_nonzero": result["test_min_region_grid_count"] > 0,
        "validation_regions_nonzero": (
            result["validation_min_region_grid_count"] > 0
        ),
        "grid_leakage_zero": result["grid_leakage_count"] == 0,
        "duplicate_assignment_zero": (
            result["duplicate_assignment_count"] == 0
        ),
        "all_months": (
            (result["train_month_count"] == 12)
            & (result["validation_month_count"] == 12)
            & (result["test_month_count"] == 12)
        ),
        "target_std_not_extreme": result["test_target_std"]
        >= median_std
        * float(acceptance["min_target_std_vs_fold_median"]),
    }
    for name, values in checks.items():
        result[f"accept_{name}"] = values
    per_fold_columns = [f"accept_{name}" for name in checks]
    result["fold_passed"] = result[per_fold_columns].all(axis=1)
    global_summary = {
        "test_target_std_median": median_std,
        "zero_baseline_rmse_mean": zero_mean,
        "zero_baseline_rmse_cv": zero_cv,
        "zero_baseline_rmse_cv_limit": float(
            acceptance["max_zero_baseline_rmse_cv"]
        ),
        "zero_baseline_rmse_cv_passed": zero_cv
        <= float(acceptance["max_zero_baseline_rmse_cv"]),
        "all_folds_passed": bool(result["fold_passed"].all()),
    }
    global_summary["audit_passed"] = bool(
        global_summary["all_folds_passed"]
        and global_summary["zero_baseline_rmse_cv_passed"]
    )
    return result, global_summary
