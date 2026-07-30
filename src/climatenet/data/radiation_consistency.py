"""Auditing helpers for ERA5-Land accumulated-radiation consistency.

The functions in this module are read-only.  They deliberately keep raw,
processed, and row-wise feature checks separate so the first stage at which a
problem appears remains auditable.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class NumericMoments:
    count: int = 0
    finite_count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    nan_count: int = 0
    positive_inf_count: int = 0
    negative_inf_count: int = 0

    def update(self, values: Iterable[float]) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        self.count += int(array.size)
        self.nan_count += int(np.isnan(array).sum())
        self.positive_inf_count += int(np.isposinf(array).sum())
        self.negative_inf_count += int(np.isneginf(array).sum())
        finite = array[np.isfinite(array)]
        if not finite.size:
            return
        self.finite_count += int(finite.size)
        self.total += float(finite.sum(dtype=np.float64))
        self.total_sq += float(np.square(finite).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))

    def summary(self) -> dict[str, float | int | None]:
        if not self.finite_count:
            mean = std = minimum = maximum = None
        else:
            mean = self.total / self.finite_count
            variance = max(
                self.total_sq / self.finite_count - mean * mean, 0.0
            )
            std = float(np.sqrt(variance))
            minimum = self.minimum
            maximum = self.maximum
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nan_count": self.nan_count,
            "positive_inf_count": self.positive_inf_count,
            "negative_inf_count": self.negative_inf_count,
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def radiation_conversions(
    raw_ssrd: np.ndarray | pd.Series,
    days_in_month: np.ndarray | pd.Series | int,
) -> dict[str, np.ndarray]:
    """Return the candidate conversions requested by the audit."""
    raw = np.asarray(raw_ssrd, dtype=np.float64)
    days = np.asarray(days_in_month, dtype=np.float64)
    return {
        "current_monthly_total_mj_m2": raw * days / 1_000_000.0,
        "no_day_multiplier_mj_m2": raw / 1_000_000.0,
        "attrs_informed_daily_mean_w_m2": raw / 86_400.0,
    }


def missing_months(times: Iterable[Any]) -> list[str]:
    periods = pd.PeriodIndex(pd.to_datetime(list(times)), freq="M")
    if periods.empty:
        return []
    observed = periods.unique().sort_values()
    expected = pd.period_range(observed.min(), observed.max(), freq="M")
    return [str(value) for value in expected.difference(observed)]


def summarize_grid_stability(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Summarize exact grid membership for small/in-memory audit frames."""
    required = ["region", "year", "month", "latitude", "longitude"]
    absent = [column for column in required if column not in records]
    if absent:
        raise ValueError(f"Grid audit missing columns: {absent}")
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    reference: dict[str, set[tuple[float, float]]] = {}
    for keys, group in records.groupby(
        ["region", "year", "month"], observed=True, sort=True
    ):
        region, year, month = keys
        coordinates = list(
            zip(group["latitude"].astype(float), group["longitude"].astype(float))
        )
        grid = set(coordinates)
        duplicate_count = len(coordinates) - len(grid)
        if region not in reference:
            reference[str(region)] = grid
        missing = len(reference[str(region)] - grid)
        added = len(grid - reference[str(region)])
        if duplicate_count or missing or added:
            warnings.append(
                f"{region} {int(year):04d}-{int(month):02d}: "
                f"duplicates={duplicate_count}, missing={missing}, added={added}"
            )
        rows.append(
            {
                "region": region,
                "year": int(year),
                "month": int(month),
                "row_count": len(group),
                "unique_grid_cells": len(grid),
                "duplicate_grid_cells": duplicate_count,
                "missing_vs_reference": missing,
                "added_vs_reference": added,
                "latitude_min": float(group["latitude"].min()),
                "latitude_max": float(group["latitude"].max()),
                "longitude_min": float(group["longitude"].min()),
                "longitude_max": float(group["longitude"].max()),
            }
        )
    return pd.DataFrame(rows), warnings


def compare_monthly_radiation(
    raw_monthly: pd.DataFrame,
    processed_monthly: pd.DataFrame,
    *,
    tolerance: float = 1e-5,
) -> pd.DataFrame:
    keys = ["region", "year", "month"]
    merged = raw_monthly[
        [*keys, "current_mean_mj_m2"]
    ].merge(
        processed_monthly[[*keys, "radiation_mean"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    merged["absolute_difference"] = (
        merged["radiation_mean"] - merged["current_mean_mj_m2"]
    ).abs()
    merged["within_tolerance"] = (
        merged["_merge"].eq("both")
        & merged["absolute_difference"].le(tolerance)
    )
    return merged


def standardized_mean_difference(
    reference: NumericMoments, comparison: NumericMoments
) -> float:
    train = reference.summary()
    test = comparison.summary()
    if train["std"] is None or test["std"] is None:
        return float("nan")
    pooled = np.sqrt((float(train["std"]) ** 2 + float(test["std"]) ** 2) / 2)
    if pooled == 0:
        return 0.0
    return (float(test["mean"]) - float(train["mean"])) / pooled


def update_grouped_moments(
    store: dict[tuple[Any, ...], NumericMoments],
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
) -> None:
    for keys, group in frame.groupby(group_columns, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for column in value_columns:
            store[(*keys, column)].update(group[column].to_numpy())


def grouped_moments_frame(
    store: dict[tuple[Any, ...], NumericMoments],
    key_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for key, moments in sorted(store.items(), key=lambda item: item[0]):
        *values, variable = key
        rows.append(
            {
                **dict(zip(key_columns, values)),
                "variable": variable,
                **moments.summary(),
            }
        )
    return pd.DataFrame(rows)
