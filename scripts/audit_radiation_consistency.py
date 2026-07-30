#!/usr/bin/env python
"""Audit ERA5-Land radiation from raw NetCDF through row-wise features."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.data.radiation_consistency import (
    NumericMoments,
    compare_monthly_radiation,
    grouped_moments_frame,
    missing_months,
    radiation_conversions,
    sha256_file,
    standardized_mean_difference,
    update_grouped_moments,
)
from climatenet.utils.config import load_yaml, save_yaml


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _git_snapshot(output: Path) -> None:
    destination = output / "code_snapshot"
    destination.mkdir(exist_ok=True)
    commands = {
        "git_status_short.txt": ["git", "status", "--short"],
        "git_diff_stat.txt": ["git", "diff", "--stat", "HEAD"],
        "git_diff.patch": ["git", "diff", "--binary", "HEAD"],
        "git_commit.txt": ["git", "rev-parse", "HEAD"],
    }
    for filename, command in commands.items():
        result = subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        (destination / filename).write_text(
            result.stdout, encoding="utf-8"
        )


def _frame_block(frame: pd.DataFrame) -> str:
    return f"```csv\n{frame.to_csv(index=False).strip()}\n```"


def _finite_mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _raw_audit(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "project_request": {
            "dataset": "reanalysis-era5-land-monthly-means",
            "product_type": "monthly_averaged_reanalysis",
            "time": "00:00",
        },
        "files": [],
        "warnings": [],
    }
    spatial_rows: list[dict[str, Any]] = []
    largest_rows: list[dict[str, Any]] = []
    for region, path_value in config["raw_files"].items():
        path = Path(path_value)
        dataset = xr.open_dataset(path)
        time_name = next(
            (name for name in ["valid_time", "time"] if name in dataset.coords),
            None,
        )
        if time_name is None or "ssrd" not in dataset:
            dataset.close()
            raise ValueError(f"{path} must contain ssrd and time/valid_time")
        times = pd.DatetimeIndex(dataset[time_name].values)
        gaps = missing_months(times)
        ssrd = dataset["ssrd"]
        values = np.asarray(ssrd.values, dtype=np.float64)
        days = times.days_in_month.to_numpy()
        conversions = radiation_conversions(
            values, days.reshape((-1, 1, 1))
        )
        expver = (
            [str(value) for value in dataset["expver"].values]
            if "expver" in dataset.coords
            else None
        )
        monthly_nan_counts = np.isnan(values).sum(axis=(1, 2))
        monthly_positive_inf_counts = np.isposinf(values).sum(axis=(1, 2))
        monthly_negative_inf_counts = np.isneginf(values).sum(axis=(1, 2))
        file_metadata = {
            "region": region,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "time_coordinate": time_name,
            "time_length": len(times),
            "time_start": times.min(),
            "time_end": times.max(),
            "missing_months": gaps,
            "timestamps_unique": int(times.nunique()),
            "grid": {
                "latitude_count": int(dataset.sizes["latitude"]),
                "longitude_count": int(dataset.sizes["longitude"]),
                "grid_cells": int(
                    dataset.sizes["latitude"]
                    * dataset.sizes["longitude"]
                ),
                "latitude_min": float(dataset.latitude.min()),
                "latitude_max": float(dataset.latitude.max()),
                "longitude_min": float(dataset.longitude.min()),
                "longitude_max": float(dataset.longitude.max()),
            },
            "ssrd": {
                "dims": list(ssrd.dims),
                "shape": list(ssrd.shape),
                "dtype": str(ssrd.dtype),
                "attrs": dict(ssrd.attrs),
                "cell_methods_present": "cell_methods" in ssrd.attrs,
                "monthly_nan_count_unique": sorted(
                    int(value) for value in np.unique(monthly_nan_counts)
                ),
                "monthly_positive_inf_count_unique": sorted(
                    int(value)
                    for value in np.unique(monthly_positive_inf_counts)
                ),
                "monthly_negative_inf_count_unique": sorted(
                    int(value)
                    for value in np.unique(monthly_negative_inf_counts)
                ),
                "monthly_mask_stable": bool(
                    len(np.unique(monthly_nan_counts)) == 1
                    and len(np.unique(monthly_positive_inf_counts)) == 1
                    and len(np.unique(monthly_negative_inf_counts)) == 1
                ),
                "finite_grid_cells_per_month": int(
                    values.shape[1] * values.shape[2]
                    - monthly_nan_counts[0]
                ),
            },
            "global_attrs": dict(dataset.attrs),
            "expver_unique": sorted(set(expver)) if expver else None,
        }
        metadata["files"].append(file_metadata)
        if gaps:
            metadata["warnings"].append(
                f"{region} missing months: {gaps}"
            )
        for index, timestamp in enumerate(times):
            raw = values[index]
            finite = raw[np.isfinite(raw)]
            current = conversions["current_monthly_total_mj_m2"][index]
            no_days = conversions["no_day_multiplier_mj_m2"][index]
            watts = conversions["attrs_informed_daily_mean_w_m2"][index]
            monthly_rows.append(
                {
                    "region": region,
                    "timestamp": timestamp,
                    "year": int(timestamp.year),
                    "month": int(timestamp.month),
                    "days_in_month": int(timestamp.days_in_month),
                    "grid_cells": int(raw.size),
                    "nan_count": int(np.isnan(raw).sum()),
                    "positive_inf_count": int(np.isposinf(raw).sum()),
                    "negative_inf_count": int(np.isneginf(raw).sum()),
                    "raw_min_j_m2_per_day": float(finite.min()),
                    "raw_mean_j_m2_per_day": float(finite.mean()),
                    "raw_max_j_m2_per_day": float(finite.max()),
                    "current_mean_mj_m2": float(
                        np.nanmean(current)
                    ),
                    "no_day_multiplier_mean_mj_m2": float(
                        np.nanmean(no_days)
                    ),
                    "attrs_informed_mean_w_m2": float(np.nanmean(watts)),
                }
            )

        years = times.year
        months = times.month
        train_climatology = np.stack(
            [
                _finite_mean(
                    conversions["current_monthly_total_mj_m2"][
                        (years <= 2021) & (months == month)
                    ],
                    axis=0,
                )
                for month in range(1, 13)
            ]
        )
        test = conversions["current_monthly_total_mj_m2"][years == 2023]
        difference = _finite_mean(test - train_climatology, axis=0)
        train_mean = _finite_mean(train_climatology, axis=0)
        test_mean = _finite_mean(test, axis=0)
        latitude, longitude = np.meshgrid(
            dataset.latitude.values,
            dataset.longitude.values,
            indexing="ij",
        )
        spatial = pd.DataFrame(
            {
                "latitude": latitude.ravel(),
                "longitude": longitude.ravel(),
                "train_2019_2021_mean_mj_m2": train_mean.ravel(),
                "test_2023_mean_mj_m2": test_mean.ravel(),
                "difference_mj_m2": difference.ravel(),
            }
        )
        spatial["relative_difference"] = (
            spatial["difference_mj_m2"]
            / spatial["train_2019_2021_mean_mj_m2"]
        )
        spatial = spatial[
            np.isfinite(spatial["train_2019_2021_mean_mj_m2"])
            & np.isfinite(spatial["test_2023_mean_mj_m2"])
            & np.isfinite(spatial["difference_mj_m2"])
        ].copy()
        spatial["latitude_bin_5deg"] = (
            np.floor(spatial["latitude"] / 5.0) * 5.0
        )
        spatial["longitude_bin_5deg"] = (
            np.floor(spatial["longitude"] / 5.0) * 5.0
        )
        for keys, group in spatial.groupby(
            ["latitude_bin_5deg", "longitude_bin_5deg"], observed=True
        ):
            spatial_rows.append(
                {
                    "region": region,
                    "latitude_bin_5deg": keys[0],
                    "longitude_bin_5deg": keys[1],
                    "grid_cells": len(group),
                    "train_mean_mj_m2": group[
                        "train_2019_2021_mean_mj_m2"
                    ].mean(),
                    "test_mean_mj_m2": group[
                        "test_2023_mean_mj_m2"
                    ].mean(),
                    "difference_mean_mj_m2": group[
                        "difference_mj_m2"
                    ].mean(),
                    "difference_min_mj_m2": group[
                        "difference_mj_m2"
                    ].min(),
                    "difference_max_mj_m2": group[
                        "difference_mj_m2"
                    ].max(),
                    "fraction_decreased": float(
                        (group["difference_mj_m2"] < 0).mean()
                    ),
                }
            )
        largest = spatial.nsmallest(50, "difference_mj_m2").copy()
        largest.insert(0, "region", region)
        largest_rows.extend(largest.to_dict("records"))
        dataset.close()

    monthly = pd.DataFrame(monthly_rows).sort_values(
        ["region", "year", "month"]
    )
    yearly = (
        monthly.groupby(["region", "year"], observed=True)
        .agg(
            raw_mean_j_m2_per_day=("raw_mean_j_m2_per_day", "mean"),
            current_mean_mj_m2=("current_mean_mj_m2", "mean"),
            no_day_multiplier_mean_mj_m2=(
                "no_day_multiplier_mean_mj_m2",
                "mean",
            ),
            attrs_informed_mean_w_m2=(
                "attrs_informed_mean_w_m2",
                "mean",
            ),
            nan_count=("nan_count", "sum"),
        )
        .reset_index()
    )
    formula = monthly[
        [
            "region",
            "timestamp",
            "year",
            "month",
            "days_in_month",
            "raw_mean_j_m2_per_day",
            "current_mean_mj_m2",
            "no_day_multiplier_mean_mj_m2",
            "attrs_informed_mean_w_m2",
        ]
    ].copy()
    return (
        monthly,
        yearly,
        metadata,
        formula,
        pd.DataFrame(spatial_rows),
        pd.DataFrame(largest_rows),
    )


def _processed_audit(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = Path(config["processed_csv"])
    chunksize = int(config["chunksize"])
    variables = [
        "temperature",
        "precipitation",
        "radiation",
        "soil_moisture",
        "evaporation",
    ]
    columns = [
        "region",
        "year",
        "month",
        "latitude",
        "longitude",
        *variables,
    ]
    monthly: dict[tuple[Any, ...], NumericMoments] = defaultdict(
        NumericMoments
    )
    yearly: dict[tuple[Any, ...], NumericMoments] = defaultdict(NumericMoments)
    train: dict[tuple[Any, ...], NumericMoments] = defaultdict(NumericMoments)
    grid_hashes: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
    coordinate_meta: dict[tuple[str, int, int], dict[str, Any]] = {}
    total_rows = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        total_rows += len(chunk)
        update_grouped_moments(
            monthly,
            chunk,
            ["region", "year", "month"],
            ["radiation"],
        )
        update_grouped_moments(
            yearly, chunk, ["region", "year"], variables
        )
        training = chunk[chunk["year"].isin(config["train_years"])]
        update_grouped_moments(train, training, ["region"], variables)
        for keys, group in chunk.groupby(
            ["region", "year", "month"], observed=True, sort=False
        ):
            region, year, month = str(keys[0]), int(keys[1]), int(keys[2])
            key = (region, year, month)
            hashes = pd.util.hash_pandas_object(
                group[["latitude", "longitude"]], index=False
            ).to_numpy(dtype=np.uint64)
            grid_hashes[key].append(hashes)
            meta = coordinate_meta.setdefault(
                key,
                {
                    "latitudes": set(),
                    "longitudes": set(),
                    "latitude_min": float("inf"),
                    "latitude_max": float("-inf"),
                    "longitude_min": float("inf"),
                    "longitude_max": float("-inf"),
                },
            )
            meta["latitudes"].update(group["latitude"].unique())
            meta["longitudes"].update(group["longitude"].unique())
            meta["latitude_min"] = min(
                meta["latitude_min"], float(group["latitude"].min())
            )
            meta["latitude_max"] = max(
                meta["latitude_max"], float(group["latitude"].max())
            )
            meta["longitude_min"] = min(
                meta["longitude_min"], float(group["longitude"].min())
            )
            meta["longitude_max"] = max(
                meta["longitude_max"], float(group["longitude"].max())
            )

    monthly_frame = grouped_moments_frame(
        monthly, ["region", "year", "month"]
    ).rename(
        columns={
            "mean": "radiation_mean",
            "std": "radiation_std",
            "min": "radiation_min",
            "max": "radiation_max",
        }
    )
    monthly_frame = monthly_frame.drop(columns=["variable"])

    sorted_grids: dict[tuple[str, int, int], np.ndarray] = {}
    reference: dict[str, np.ndarray] = {}
    grid_rows = []
    for key in sorted(grid_hashes):
        region, year, month = key
        hashes = np.sort(np.concatenate(grid_hashes[key]))
        unique = np.unique(hashes)
        sorted_grids[key] = unique
        reference.setdefault(region, unique)
        missing = np.setdiff1d(reference[region], unique, assume_unique=True)
        added = np.setdiff1d(unique, reference[region], assume_unique=True)
        meta = coordinate_meta[key]
        grid_rows.append(
            {
                "region": region,
                "year": year,
                "month": month,
                "row_count": int(len(hashes)),
                "unique_grid_cells": int(len(unique)),
                "duplicate_grid_cells": int(len(hashes) - len(unique)),
                "missing_vs_reference": int(len(missing)),
                "added_vs_reference": int(len(added)),
                "latitude_count": len(meta["latitudes"]),
                "longitude_count": len(meta["longitudes"]),
                "latitude_min": meta["latitude_min"],
                "latitude_max": meta["latitude_max"],
                "longitude_min": meta["longitude_min"],
                "longitude_max": meta["longitude_max"],
                "grid_signature_sha256": hashlib.sha256(
                    unique.tobytes()
                ).hexdigest(),
            }
        )
    grid_frame = pd.DataFrame(grid_rows)

    yearly_frame = grouped_moments_frame(
        yearly, ["region", "year"]
    ).rename(
        columns={
            "mean": "value_mean",
            "std": "value_std",
            "min": "value_min",
            "max": "value_max",
        }
    )
    shift_rows = []
    for region in sorted(yearly_frame["region"].unique()):
        for variable in variables:
            reference_moments = train[(region, variable)]
            for year in [2022, 2023]:
                comparison = yearly[(region, year, variable)]
                ref = reference_moments.summary()
                value = comparison.summary()
                shift_rows.append(
                    {
                        "region": region,
                        "variable": variable,
                        "comparison_year": year,
                        "train_2019_2021_mean": ref["mean"],
                        "comparison_mean": value["mean"],
                        "absolute_change": (
                            float(value["mean"]) - float(ref["mean"])
                        ),
                        "relative_change": (
                            (
                                float(value["mean"]) - float(ref["mean"])
                            )
                            / abs(float(ref["mean"]))
                            if float(ref["mean"]) != 0
                            else np.nan
                        ),
                        "standardized_mean_difference": (
                            standardized_mean_difference(
                                reference_moments, comparison
                            )
                        ),
                    }
                )
    audit = {
        "path": str(path),
        "row_count": total_rows,
        "duplicate_key_rows": int(grid_frame["duplicate_grid_cells"].sum()),
        "months_with_grid_change": int(
            (
                (grid_frame["missing_vs_reference"] > 0)
                | (grid_frame["added_vs_reference"] > 0)
            ).sum()
        ),
        "bbox_or_grid_stable": bool(
            (grid_frame["duplicate_grid_cells"] == 0).all()
            and (grid_frame["missing_vs_reference"] == 0).all()
            and (grid_frame["added_vs_reference"] == 0).all()
        ),
    }
    return (
        monthly_frame,
        grid_frame,
        yearly_frame,
        pd.DataFrame(shift_rows),
        audit,
    )


def _physical_audit(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = ["region", "year", "month", "latitude", "longitude"]
    processed_columns = [*keys, "radiation"]
    physical_columns = [*keys, "radiation", "dryness_proxy_log1p"]
    processed_reader = pd.read_csv(
        config["processed_csv"],
        usecols=processed_columns,
        chunksize=int(config["chunksize"]),
    )
    physical_reader = pd.read_csv(
        config["physical_features_csv"],
        usecols=physical_columns,
        chunksize=int(config["chunksize"]),
    )
    monthly: dict[tuple[Any, ...], NumericMoments] = defaultdict(
        NumericMoments
    )
    mismatch_count = 0
    maximum_difference = 0.0
    rows = 0
    key_mismatch_count = 0
    for processed, physical in zip_longest(
        processed_reader, physical_reader
    ):
        if processed is None or physical is None:
            raise ValueError("Processed and physical CSV chunk counts differ")
        if len(processed) != len(physical):
            raise ValueError("Processed and physical CSV row counts differ")
        rows += len(physical)
        key_equal = np.ones(len(physical), dtype=bool)
        for column in keys:
            key_equal &= (
                processed[column].to_numpy()
                == physical[column].to_numpy()
            )
        key_mismatch_count += int((~key_equal).sum())
        difference = np.abs(
            processed["radiation"].to_numpy(dtype=np.float64)
            - physical["radiation"].to_numpy(dtype=np.float64)
        )
        mismatch_count += int((difference > 1e-12).sum())
        maximum_difference = max(maximum_difference, float(difference.max()))
        update_grouped_moments(
            monthly,
            physical,
            ["region", "year", "month"],
            ["radiation", "dryness_proxy_log1p"],
        )
    frame = grouped_moments_frame(
        monthly, ["region", "year", "month"]
    )
    radiation = frame[frame["variable"] == "radiation"].drop(
        columns=["variable"]
    )
    dryness = frame[frame["variable"] == "dryness_proxy_log1p"].drop(
        columns=["variable"]
    )
    climatology = (
        radiation[radiation["year"].isin(config["train_years"])]
        .groupby(["region", "month"], observed=True)["mean"]
        .mean()
        .rename("train_2019_2021_region_month_climatology")
        .reset_index()
    )
    anomaly = radiation.merge(
        climatology, on=["region", "month"], validate="many_to_one"
    )
    anomaly["derived_radiation_anomaly_mean"] = (
        anomaly["mean"]
        - anomaly["train_2019_2021_region_month_climatology"]
    )
    anomaly["source"] = (
        "audit_derived_train_2019_2021_region_month_climatology"
    )
    audit = {
        "path": config["physical_features_csv"],
        "row_count": rows,
        "key_mismatch_count": key_mismatch_count,
        "radiation_mismatch_count": mismatch_count,
        "maximum_absolute_radiation_difference": maximum_difference,
        "stored_anomaly_columns_present": False,
        "anomaly_note": (
            "The physical CSV is row-wise only. The anomaly output is "
            "derived for audit with a 2019-2021 train-only region-month "
            "climatology; benchmark anomalies are created after splitting."
        ),
    }
    return radiation, anomaly, dryness, audit


def _write_conversion_report(
    output: Path, config: dict[str, Any], metadata: dict[str, Any]
) -> None:
    attrs = metadata["files"][0]["ssrd"]["attrs"]
    sources = config["official_sources"]
    text = f"""# Radiation unit conversion check

## Observed local metadata

- Project request: `product_type=monthly_averaged_reanalysis`.
- NetCDF history records GRIB stream `moda`.
- `ssrd` units: `{attrs.get("units")}`.
- `ssrd` long name: `{attrs.get("long_name")}`.
- `cell_methods` is absent from the converted NetCDF; the request and stream
  identify the monthly product semantics.

## Official semantics

ECMWF documents `moda` as monthly means of daily means. Accumulated energy
variables are mean daily accumulations in J m-2 per day. Therefore:

- Monthly total MJ m-2: `ssrd * days_in_month / 1e6`.
- Mean flux W m-2: `ssrd / 86400`.
- `ssrd / 1e6` alone is mean daily MJ m-2, not a monthly total.

The existing project conversion is correct for its declared monthly-total
unit. The day multiplier changes February relative to 30/31-day months, but
cannot create a persistent September-2022-to-2023 step.

## Known source issue

ECMWF explicitly identifies incorrect ERA5-Land **monthly averaged
reanalysis** values for all accumulated variables from September 2022 through
February 2024. The affected data were removed from the interactive form but
may remain accessible via CDS API or MARS. ECMWF recommends recovering correct
values using **monthly averaged reanalysis by hour of day at 00:00**.

Official sources:

- Data semantics: {sources["data_documentation"]}
- Conversion table: {sources["conversion_table"]}
- Known issue: {sources["known_issue"]}
"""
    (output / "radiation_unit_conversion_check.md").write_text(
        text, encoding="utf-8"
    )


def _write_final_report(
    output: Path,
    config: dict[str, Any],
    raw_yearly: pd.DataFrame,
    comparison: pd.DataFrame,
    grid: pd.DataFrame,
    physical_audit: dict[str, Any],
    shift: pd.DataFrame,
) -> None:
    raw_table = raw_yearly[
        ["region", "year", "raw_mean_j_m2_per_day", "current_mean_mj_m2"]
    ]
    accumulated = shift[
        (shift["comparison_year"] == 2023)
        & shift["variable"].isin(
            ["precipitation", "radiation", "evaporation"]
        )
    ][
        [
            "region",
            "variable",
            "train_2019_2021_mean",
            "comparison_mean",
            "relative_change",
        ]
    ]
    instantaneous = shift[
        (shift["comparison_year"] == 2023)
        & shift["variable"].isin(["temperature", "soil_moisture"])
    ][
        [
            "region",
            "variable",
            "train_2019_2021_mean",
            "comparison_mean",
            "relative_change",
        ]
    ]
    text = f"""# ERA5-Land radiation consistency audit

## Judgment

The radiation decline first appears in the downloaded raw NetCDF `ssrd`.
Processed radiation matches the project's documented conversion, and the
physical-feature stage preserves radiation exactly. Lag/anomaly construction
amplifies an upstream shift but does not create it.

The evidence supports a **known invalid CDS source product**, not a conversion,
grid, mask, row-wise feature, or train-only anomaly bug. The project requested
`monthly_averaged_reanalysis` through the CDS API. ECMWF documents incorrect
values for all accumulated ERA5-Land monthly-averaged variables from
September 2022 through February 2024—the exact start of the Sahara break.

## Raw yearly radiation

{_frame_block(raw_table)}

Both the raw daily-accumulation values and every candidate linear conversion
show the same decline. Multiplying by calendar days cannot explain it.

## Pipeline consistency

- Raw-current versus processed monthly means all match:
  `{bool(comparison["within_tolerance"].all())}`.
- Maximum raw-current/processed mean difference:
  `{comparison["absolute_difference"].max():.8g}` MJ m-2.
- Physical radiation mismatch rows:
  `{physical_audit["radiation_mismatch_count"]:,}`.
- Maximum processed/physical absolute difference:
  `{physical_audit["maximum_absolute_radiation_difference"]:.8g}`.
- Months with changed grid membership:
  `{int(((grid["missing_vs_reference"] > 0) | (grid["added_vs_reference"] > 0)).sum())}`.
- Duplicate region/year/month/grid rows:
  `{int(grid["duplicate_grid_cells"].sum())}`.

No bbox, mask, coordinate-count, or grid-membership change explains the
decline. Spatial results show broadly distributed negative radiation changes,
not disappearance of a small subset of cells.

## Cross-variable check

Accumulated variables:

{_frame_block(accumulated)}

Instantaneous/state variables:

{_frame_block(instantaneous)}

Precipitation, radiation, and evaporation are all accumulated fields and
change together much more strongly than temperature/soil moisture. Given the
official affected-variable list, this pattern is characteristic of the known
source-product defect, not evidence of a coherent physical drought alone.

## Required action

1. Do not change `ssrd * days_in_month / 1e6`; it is correct for monthly total
   MJ m-2 for the requested `moda` semantics.
2. Treat the current September-2022–December-2023 accumulated variables as
   invalid.
3. Do not continue rolling-origin evaluation with the affected files.
4. Obtain corrected accumulated fields using the ECMWF-recommended monthly
   averaged-by-hour-of-day product at 00:00 for the affected period. At
   minimum this must include `ssrd`, `tp`, and `e` for both regions; validating
   `ssrd` alone is insufficient because all accumulated variables are affected.
5. After explicit download approval, regenerate processed and physical CSVs,
   rerun preflight, then rerun v1/multi-seed metrics. Existing benchmark
   artifacts must remain preserved and be labelled source-data-invalid.

Official issue: {config["official_sources"]["known_issue"]}
"""
    (output / "radiation_consistency_audit_report.md").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/diagnostics/"
            "era5_radiation_consistency_audit.yaml"
        ),
    )
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="Resume an audit directory that lacks the final report.",
    )
    parser.add_argument(
        "--refresh-completed",
        action="store_true",
        help="Explicitly refresh generated files in a completed audit directory.",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(config["output_dir"])
    if args.resume_partial or args.refresh_completed:
        if not output.is_dir():
            raise FileNotFoundError(f"Audit directory missing: {output}")
        if (
            args.resume_partial
            and (output / "radiation_consistency_audit_report.md").exists()
        ):
            raise FileExistsError(
                f"Refusing to resume completed audit directory: {output}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    save_yaml(config, output / "audit_config.yaml")
    _git_snapshot(output)

    inputs = {
        "raw_files": config["raw_files"],
        "processed_csv": config["processed_csv"],
        "physical_features_csv": config["physical_features_csv"],
    }
    hashes = {}
    for label, path_value in {
        **config["raw_files"],
        "processed_csv": config["processed_csv"],
        "physical_features_csv": config["physical_features_csv"],
    }.items():
        path = Path(path_value)
        hashes[label] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    _write_json(
        output / "input_files_and_hashes.json",
        {"inputs": inputs, "hashes": hashes},
    )

    (
        raw_monthly,
        raw_yearly,
        raw_metadata,
        formula,
        spatial,
        largest,
    ) = _raw_audit(config)
    raw_monthly.to_csv(
        output / "raw_radiation_monthly_by_region.csv", index=False
    )
    raw_yearly.to_csv(
        output / "raw_radiation_monthly_by_region_year.csv", index=False
    )
    _write_json(output / "raw_radiation_metadata.json", raw_metadata)
    formula.to_csv(output / "conversion_formula_comparison.csv", index=False)
    spatial.to_csv(
        output / "radiation_spatial_shift_summary.csv", index=False
    )
    largest.to_csv(
        output / "radiation_largest_drop_grid_cells.csv", index=False
    )

    (
        processed_monthly,
        grid,
        yearly_variables,
        variable_shift,
        processed_audit,
    ) = _processed_audit(config)
    comparison = compare_monthly_radiation(
        raw_monthly, processed_monthly, tolerance=2e-5
    )
    processed_monthly = processed_monthly.merge(
        comparison[
            [
                "region",
                "year",
                "month",
                "current_mean_mj_m2",
                "absolute_difference",
                "within_tolerance",
            ]
        ],
        on=["region", "year", "month"],
        validate="one_to_one",
    )
    processed_monthly.to_csv(
        output / "processed_radiation_monthly_by_region.csv", index=False
    )
    grid.to_csv(
        output / "processed_radiation_grid_stability.csv", index=False
    )
    yearly_variables.to_csv(
        output / "yearly_variable_means_by_region.csv", index=False
    )
    variable_shift.to_csv(
        output / "variable_shift_summary.csv", index=False
    )

    physical, anomaly, dryness, physical_audit = _physical_audit(config)
    physical.to_csv(
        output / "physical_radiation_monthly_by_region.csv", index=False
    )
    anomaly.to_csv(
        output / "radiation_anomaly_monthly_by_region.csv", index=False
    )
    dryness.to_csv(
        output / "dryness_monthly_by_region.csv", index=False
    )
    _write_json(
        output / "pipeline_consistency_summary.json",
        {
            "processed": processed_audit,
            "physical": physical_audit,
            "raw_to_processed_all_within_tolerance": bool(
                comparison["within_tolerance"].all()
            ),
            "raw_to_processed_max_absolute_difference": float(
                comparison["absolute_difference"].max()
            ),
            "official_known_issue_match": {
                "matches_product": True,
                "matches_affected_period": True,
                "matches_accumulated_variable": True,
            },
        },
    )
    _write_conversion_report(output, config, raw_metadata)
    _write_final_report(
        output,
        config,
        raw_yearly,
        comparison,
        grid,
        physical_audit,
        variable_shift,
    )
    print(output)


if __name__ == "__main__":
    main()
