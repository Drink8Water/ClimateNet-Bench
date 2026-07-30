"""Readiness audit for bounded ERA5-Land inputs.

The audit is intentionally separate from benchmark preprocessing: it inspects
raw NetCDF metadata and values, applies the same unit conversion used by the
tabular converter, and estimates lag availability without fitting anomalies or
any train/validation/test statistic.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.data.era5_preprocess import (
    ERA5_RENAME_MAP,
    PROJECT_SCHEMA_COLUMNS,
    infer_region_from_filename,
    open_era5_subset,
)
from climatenet.utils.paths import resolve_project_path

EXPECTED_UNIT_TOKENS: dict[str, tuple[str, ...]] = {
    "t2m": ("k",),
    "tp": ("m",),
    "ssrd": ("j", "m"),
    "e": ("m", "water"),
    "swvl1": ("m",),
    "u10": ("m", "s"),
    "v10": ("m", "s"),
}

UNIT_CONVERSIONS = {
    "temperature": "K -> degC: value - 273.15",
    "precipitation": (
        "monthly-mean daily accumulation m/day -> mm/month: "
        "value * 1000 * days_in_month"
    ),
    "radiation": (
        "monthly-mean daily accumulation J m-2/day -> MJ m-2/month: "
        "value / 1e6 * days_in_month"
    ),
    "evaporation": (
        "ERA5 upward evaporation is normally negative; convert to positive "
        "mm/month: -value * 1000 * days_in_month"
    ),
    "soil_moisture": "m3 m-3, unchanged",
    "u_wind": "m s-1, unchanged",
    "v_wind": "m s-1, unchanged",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def _numeric_summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values)
    numeric = pd.to_numeric(pd.Series(array.reshape(-1)), errors="coerce")
    finite_mask = np.isfinite(numeric)
    finite = numeric[finite_mask]
    return {
        "count": int(len(numeric)),
        "missing_count": int(numeric.isna().sum()),
        "positive_inf_count": int(np.isposinf(numeric).sum()),
        "negative_inf_count": int(np.isneginf(numeric).sum()),
        "non_finite_count": int((~finite_mask).sum()),
        "min": float(finite.min()) if len(finite) else None,
        "mean": float(finite.mean()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
    }


def _missing_months(times: pd.DatetimeIndex) -> list[str]:
    if times.empty:
        return []
    observed = pd.PeriodIndex(times, freq="M").unique().sort_values()
    expected = pd.period_range(observed.min(), observed.max(), freq="M")
    return [str(period) for period in expected.difference(observed)]


def _converted_arrays(
    dataset: Any,
    times: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    """Apply the production unit formulas without materialising a DataFrame."""
    days = times.days_in_month.to_numpy(dtype=float).reshape((-1, 1, 1))
    return {
        "temperature": np.asarray(dataset["t2m"].values) - 273.15,
        "precipitation": np.asarray(dataset["tp"].values) * 1000.0 * days,
        "radiation": np.asarray(dataset["ssrd"].values) / 1_000_000.0 * days,
        "soil_moisture": np.asarray(dataset["swvl1"].values),
        "u_wind": np.asarray(dataset["u10"].values),
        "v_wind": np.asarray(dataset["v10"].values),
        "evaporation": -np.asarray(dataset["e"].values) * 1000.0 * days,
    }


def _combine_numeric_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    count = sum(int(summary["count"]) for summary in summaries)
    finite_count = sum(
        int(summary["count"]) - int(summary["non_finite_count"])
        for summary in summaries
    )
    weighted_sum = sum(
        float(summary["mean"]) * (
            int(summary["count"]) - int(summary["non_finite_count"])
        )
        for summary in summaries
        if summary["mean"] is not None
    )
    finite_minima = [
        float(summary["min"])
        for summary in summaries
        if summary["min"] is not None
    ]
    finite_maxima = [
        float(summary["max"])
        for summary in summaries
        if summary["max"] is not None
    ]
    return {
        "count": count,
        "missing_count": sum(
            int(summary["missing_count"]) for summary in summaries
        ),
        "positive_inf_count": sum(
            int(summary["positive_inf_count"]) for summary in summaries
        ),
        "negative_inf_count": sum(
            int(summary["negative_inf_count"]) for summary in summaries
        ),
        "non_finite_count": sum(
            int(summary["non_finite_count"]) for summary in summaries
        ),
        "min": min(finite_minima) if finite_minima else None,
        "mean": weighted_sum / finite_count if finite_count else None,
        "max": max(finite_maxima) if finite_maxima else None,
    }


def _period_lag_count(times: pd.DatetimeIndex, input_window: int) -> tuple[int, int]:
    periods = pd.PeriodIndex(times, freq="M").unique().sort_values()
    observed = set(periods)
    available = 0
    rejected = 0
    for target_index, target in enumerate(periods):
        if target_index < input_window:
            continue
        if all((target - lag) in observed for lag in range(1, input_window + 1)):
            available += 1
        else:
            rejected += 1
    return available, rejected


def estimate_full_artifact_storage(
    converted_rows: int,
    lag_samples: int,
    *,
    task_count: int | None = None,
    test_fraction: float = 0.15,
) -> dict[str, Any]:
    """Return conservative CSV/artifact byte estimates for preflight planning."""
    processed_low = converted_rows * 75
    processed_high = converted_rows * 130
    prediction_rows_per_task = int(round(lag_samples * test_fraction))
    effective_tasks = int(task_count or 1)
    predictions_low = prediction_rows_per_task * effective_tasks * 120
    predictions_high = prediction_rows_per_task * effective_tasks * 220
    return {
        "method": (
            "range estimate from ClimateNet CSV schemas; actual size depends "
            "on float precision, split test ratios and task matrix"
        ),
        "processed_csv_bytes": {
            "low": processed_low,
            "high": processed_high,
        },
        "forecasting_samples": lag_samples,
        "assumed_test_fraction": test_fraction,
        "task_count": task_count,
        "prediction_rows_per_task": prediction_rows_per_task,
        "predictions_and_metrics_bytes": {
            "low": predictions_low,
            "high": predictions_high,
        },
    }


def estimate_benchmark_task_count(config: dict[str, Any]) -> int:
    """Estimate the configured task matrix without generating data splits."""
    split_instances = 0
    for protocol in config.get("split_protocols", []):
        if protocol == "region_transfer":
            split_instances += len(config.get("region_transfer_pairs") or [])
        elif protocol == "climate_zone_transfer":
            split_instances += len(config.get("climate_zone_pairs") or [])
        else:
            split_instances += 1
    feature_set_count = len(config.get("feature_sets", {}))
    tasks_per_split = 0
    for model in config.get("models", []):
        name = model if isinstance(model, str) else model.get("name", "")
        tasks_per_split += (
            1 if name in {"climatology", "persistence"} else feature_set_count
        )
    return split_instances * tasks_per_split


def estimate_lag_samples(
    records: pd.DataFrame,
    input_window: int,
) -> dict[str, int]:
    """Estimate usable samples using truly consecutive calendar windows."""
    required = ["region", "latitude", "longitude", "year", "month"]
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(
            f"Cannot estimate lag samples; missing columns: {missing}"
        )
    available = 0
    rejected_nonconsecutive = 0
    for _, group in records.groupby(["region", "latitude", "longitude"]):
        periods = pd.PeriodIndex(
            group["year"].astype(int).astype(str)
            + "-"
            + group["month"].astype(int).astype(str).str.zfill(2),
            freq="M",
        ).unique().sort_values()
        observed = set(periods)
        for target_index, target in enumerate(periods):
            if target_index < input_window:
                continue
            required_history = {
                target - lag for lag in range(1, input_window + 1)
            }
            if required_history.issubset(observed):
                available += 1
            else:
                rejected_nonconsecutive += 1
    return {
        "input_window": int(input_window),
        "estimated_available_samples": int(available),
        "rejected_nonconsecutive_targets": int(rejected_nonconsecutive),
    }


def audit_era5_files(
    paths: list[str | Path],
    *,
    start: str | None = None,
    end: str | None = None,
    bbox: dict[str, list[float]] | None = None,
    max_grid_cells: int | None = None,
    max_total_bytes: int = 512 * 1024 * 1024,
    input_window: int = 6,
    region: str | None = None,
    task_count: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit ERA5 files and optionally save a JSON report.

    Statistics are accumulated per file and variable, so a full preflight
    never needs to construct the multi-million-row tabular dataset first.
    """
    if not paths:
        raise ValueError("At least one ERA5 NetCDF path is required")
    resolved_paths = [resolve_project_path(path) for path in paths]
    missing_paths = [str(path) for path in resolved_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"ERA5 input files do not exist: {missing_paths}"
        )
    total_bytes = sum(path.stat().st_size for path in resolved_paths)
    if total_bytes > max_total_bytes:
        raise ValueError(
            f"ERA5 inputs total {total_bytes:,} bytes, exceeding the audit "
            f"guardrail max_total_bytes={max_total_bytes:,}. Narrow the file "
            "list or explicitly raise the limit after reviewing I/O cost."
        )

    warnings: list[str] = []
    blocking_issues: list[str] = []

    def record_issue(message: str, *, blocking: bool = True) -> None:
        warnings.append(message)
        if blocking:
            blocking_issues.append(message)

    file_reports: list[dict[str, Any]] = []
    all_times: list[pd.Timestamp] = []
    converted_summary_parts: dict[str, list[dict[str, Any]]] = {
        column: []
        for column in [
            "temperature",
            "precipitation",
            "radiation",
            "soil_moisture",
            "u_wind",
            "v_wind",
            "evaporation",
        ]
    }
    converted_prefilter_summary_parts: dict[str, list[dict[str, Any]]] = {
        column: [] for column in converted_summary_parts
    }
    calendar_counts: list[dict[str, Any]] = []
    year_month_counts: list[dict[str, Any]] = []
    monthly_grid_counts: list[dict[str, Any]] = []
    grid_stability: dict[str, Any] = {}
    physical_risks: dict[str, Any] = {}
    total_converted_rows = 0
    total_lag_samples = 0
    total_rejected_lag_targets = 0
    evaporation_negative = 0
    evaporation_positive = 0
    evaporation_zero = 0
    total_fully_missing_rows = 0
    total_partially_invalid_rows = 0
    total_coordinate_rows = 0
    for path in resolved_paths:
        resolved_region = region or infer_region_from_filename(path)
        dataset = open_era5_subset(
            path,
            start=start,
            end=end,
            bbox=bbox,
            max_grid_cells=max_grid_cells,
        )
        time_name = next(
            name for name in ["valid_time", "time"] if name in dataset.coords
        )
        times = pd.DatetimeIndex(dataset[time_name].values)
        all_times.extend(pd.Timestamp(value) for value in times)
        missing_months = _missing_months(times)
        if missing_months:
            record_issue(
                f"{path.name} has non-continuous monthly coverage; "
                f"missing months: {missing_months}"
            )

        variables: dict[str, Any] = {}
        for raw_name, project_name in ERA5_RENAME_MAP.items():
            variable = dataset[raw_name]
            units = str(variable.attrs.get("units", "unknown"))
            unit_tokens = EXPECTED_UNIT_TOKENS[raw_name]
            unit_ok = all(token in units.casefold() for token in unit_tokens)
            if not unit_ok:
                record_issue(
                    f"{path.name}:{raw_name} units {units!r} do not match "
                    f"expected tokens {unit_tokens}"
                )
            variables[raw_name] = {
                "project_column": project_name,
                "long_name": str(variable.attrs.get("long_name", "")),
                "units": units,
                "unit_check_passed": unit_ok,
                "raw_summary": _numeric_summary(variable.values),
                "conversion": UNIT_CONVERSIONS[project_name],
            }

        latitude = np.asarray(dataset["latitude"].values, dtype=float)
        longitude = np.asarray(dataset["longitude"].values, dtype=float)
        grid_cells = int(len(latitude) * len(longitude))
        converted_arrays = _converted_arrays(dataset, times)
        finite_masks = [
            np.isfinite(values) for values in converted_arrays.values()
        ]
        missing_masks = [
            np.isnan(values) for values in converted_arrays.values()
        ]
        valid_row_mask = np.logical_and.reduce(finite_masks)
        fully_missing_row_mask = np.logical_and.reduce(missing_masks)
        partially_invalid_row_mask = (
            ~valid_row_mask & ~fully_missing_row_mask
        )
        fully_missing_rows = int(fully_missing_row_mask.sum())
        partially_invalid_rows = int(partially_invalid_row_mask.sum())
        coordinate_rows = int(valid_row_mask.size)
        retained_rows = int(valid_row_mask.sum())
        total_fully_missing_rows += fully_missing_rows
        total_partially_invalid_rows += partially_invalid_rows
        total_coordinate_rows += coordinate_rows
        if partially_invalid_rows:
            record_issue(
                f"{resolved_region} contains {partially_invalid_rows:,} rows "
                "with partial NaN/Inf values; these are not a uniform land/sea "
                "mask and require investigation."
            )
        converted_before_filter = {
            column: _numeric_summary(values)
            for column, values in converted_arrays.items()
        }
        converted_for_file = {
            column: _numeric_summary(values[valid_row_mask])
            for column, values in converted_arrays.items()
        }
        for column, summary in converted_for_file.items():
            converted_summary_parts[column].append(summary)
            converted_prefilter_summary_parts[column].append(
                converted_before_filter[column]
            )

        finite_evaporation = converted_arrays["evaporation"][valid_row_mask]
        evaporation_negative += int((finite_evaporation < 0).sum())
        evaporation_positive += int((finite_evaporation > 0).sum())
        evaporation_zero += int((finite_evaporation == 0).sum())

        dryness_all = converted_arrays["radiation"] / (
            converted_arrays["precipitation"] + 1e-6
        )
        dryness = dryness_all[valid_row_mask]
        finite_dryness = dryness[np.isfinite(dryness)]
        near_zero_precipitation = int(
            (
                np.abs(converted_arrays["precipitation"][valid_row_mask])
                < 0.01
            ).sum()
        )
        physical_risks[resolved_region] = {
            "dryness_proxy_formula": "radiation / (precipitation + 1e-6)",
            "non_finite_count": int((~np.isfinite(dryness)).sum()),
            "near_zero_precipitation_rows_below_0_01_mm": (
                near_zero_precipitation
            ),
            "min": float(finite_dryness.min()) if finite_dryness.size else None,
            "mean": float(finite_dryness.mean()) if finite_dryness.size else None,
            "p99": (
                float(np.quantile(finite_dryness, 0.99))
                if finite_dryness.size else None
            ),
            "p999": (
                float(np.quantile(finite_dryness, 0.999))
                if finite_dryness.size else None
            ),
            "max": float(finite_dryness.max()) if finite_dryness.size else None,
        }
        if physical_risks[resolved_region]["non_finite_count"]:
            record_issue(
                f"{resolved_region} dryness_proxy contains "
                f"{physical_risks[resolved_region]['non_finite_count']:,} "
                "non-finite values."
            )
        if near_zero_precipitation:
            record_issue(
                f"{resolved_region} dryness_proxy has "
                f"{near_zero_precipitation:,} rows below 0.01 mm precipitation; "
                "review its extreme-value distribution before full training.",
                blocking=False,
            )

        month_counts_for_region: dict[int, int] = {}
        monthly_retained_counts: list[int] = []
        for time_index, timestamp in enumerate(times):
            year = int(timestamp.year)
            month = int(timestamp.month)
            monthly_retained = int(valid_row_mask[time_index].sum())
            monthly_retained_counts.append(monthly_retained)
            year_month_counts.append(
                {
                    "region": resolved_region,
                    "year": year,
                    "month": month,
                    "sample_count": monthly_retained,
                }
            )
            monthly_grid_counts.append(
                {
                    "region": resolved_region,
                    "year": year,
                    "month": month,
                    "grid_cell_count": monthly_retained,
                }
            )
            month_counts_for_region[month] = (
                month_counts_for_region.get(month, 0) + monthly_retained
            )
        calendar_counts.extend(
            {
                "region": resolved_region,
                "month": month,
                "sample_count": count,
            }
            for month, count in sorted(month_counts_for_region.items())
        )
        unique_grid_counts = sorted(
            {
                row["grid_cell_count"]
                for row in monthly_grid_counts
                if row["region"] == resolved_region
            }
        )
        grid_stability[resolved_region] = {
            "stable": len(unique_grid_counts) == 1,
            "unique_monthly_grid_cell_counts": unique_grid_counts,
        }
        if len(unique_grid_counts) != 1:
            record_issue(
                f"{resolved_region} monthly grid-cell count is not stable: "
                f"{unique_grid_counts}"
            )

        available_samples = 0
        rejected_samples = 0
        for target_index in range(input_window, len(times)):
            history_valid = np.all(
                valid_row_mask[target_index - input_window:target_index],
                axis=0,
            )
            target_valid = valid_row_mask[target_index]
            available_samples += int((history_valid & target_valid).sum())
            rejected_samples += int((target_valid & ~history_valid).sum())
        total_lag_samples += available_samples
        total_rejected_lag_targets += rejected_samples
        total_converted_rows += retained_rows
        file_reports.append(
            {
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "available_variables": sorted(str(name) for name in dataset.data_vars),
                "required_variables": sorted(ERA5_RENAME_MAP),
                "coordinates": sorted(str(name) for name in dataset.coords),
                "time_coordinate": time_name,
                "time_range": {
                    "start": times.min().isoformat(),
                    "end": times.max().isoformat(),
                    "month_count": int(len(times)),
                    "missing_months": missing_months,
                },
                "latitude_range": {
                    "min": float(latitude.min()),
                    "max": float(latitude.max()),
                    "count": int(len(latitude)),
                },
                "longitude_range": {
                    "min": float(longitude.min()),
                    "max": float(longitude.max()),
                    "count": int(len(longitude)),
                },
                "grid_cell_count": grid_cells,
                "retained_grid_cell_count": int(
                    np.any(valid_row_mask, axis=0).sum()
                ),
                "region_field_in_netcdf": "region" in dataset,
                "region": resolved_region,
                "region_mapping": (
                    "explicit CLI/config value" if region is not None
                    else "inferred from controlled filename"
                ),
                "climate_zone_field_in_netcdf": (
                    "climate_zone" in dataset or "climate_type" in dataset
                ),
                "climate_zone_behavior": (
                    "optional in raw data; forecasting samples map registered "
                    "region names through RegionRegistry, otherwise 'unknown'"
                ),
                "variables": variables,
                "row_filter": {
                    "coordinate_rows": coordinate_rows,
                    "fully_missing_all_variables_rows": fully_missing_rows,
                    "partially_invalid_rows": partially_invalid_rows,
                    "retained_rows": retained_rows,
                    "policy": (
                        "uniform all-variable missing rows are treated as "
                        "land/sea mask; partial NaN/Inf rows are blocking"
                    ),
                },
                "converted_variable_summary_before_row_filter": (
                    converted_before_filter
                ),
                "converted_variable_summary": converted_for_file,
            }
        )
        dataset.close()
    converted_summaries = {
        column: _combine_numeric_summaries(summaries)
        for column, summaries in converted_summary_parts.items()
    }
    converted_prefilter_summaries = {
        column: _combine_numeric_summaries(summaries)
        for column, summaries in converted_prefilter_summary_parts.items()
    }
    non_finite_total = sum(
        summary["non_finite_count"]
        for summary in converted_summaries.values()
    )
    if non_finite_total:
        record_issue(
            f"Converted records contain {non_finite_total:,} non-finite "
            "variable values; formal benchmark preprocessing will reject them."
        )

    evaporation_sign = {
        "target_present": True,
        "direction": "positive values mean evaporation loss from land",
        "conversion_applied": UNIT_CONVERSIONS["evaporation"],
        "negative_count_after_conversion": evaporation_negative,
        "positive_count_after_conversion": evaporation_positive,
        "zero_count_after_conversion": evaporation_zero,
    }
    if evaporation_sign["negative_count_after_conversion"]:
        record_issue(
            "Converted evaporation contains a small number of negative values "
            "(possible condensation/dew after sign inversion); review their "
            "range before full training.",
            blocking=False,
        )

    unique_times = pd.DatetimeIndex(all_times).unique().sort_values()
    overall_missing_months = _missing_months(unique_times)
    if overall_missing_months:
        record_issue(
            f"Combined inputs have non-continuous monthly coverage: "
            f"{overall_missing_months}"
        )

    report = {
        "audit_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "warning" if blocking_issues else "ready",
        "audit_scope": "bounded_dry_run" if bbox is not None else "full_preflight",
        "dry_run_only": bbox is not None,
        "input_files": file_reports,
        "input_total_bytes": int(total_bytes),
        "subset": {
            "start": start,
            "end": end,
            "bbox": bbox,
            "max_grid_cells": max_grid_cells,
        },
        "converted_schema_columns": PROJECT_SCHEMA_COLUMNS,
        "converted_row_count": int(total_converted_rows),
        "row_filter_summary": {
            "coordinate_rows": int(total_coordinate_rows),
            "fully_missing_all_variables_rows": int(
                total_fully_missing_rows
            ),
            "partially_invalid_rows": int(total_partially_invalid_rows),
            "retained_rows": int(total_converted_rows),
        },
        "converted_variable_summary_before_row_filter": (
            converted_prefilter_summaries
        ),
        "converted_variable_summary": converted_summaries,
        "evaporation_target": evaporation_sign,
        "physical_feature_risk": physical_risks,
        "monthly_coverage": {
            "start": unique_times.min().isoformat(),
            "end": unique_times.max().isoformat(),
            "observed_month_count": int(len(unique_times)),
            "missing_months": overall_missing_months,
            "continuous": not overall_missing_months,
        },
        "region_calendar_month_sample_counts": [
            {key: _json_value(value) for key, value in row.items()}
            for row in calendar_counts
        ],
        "region_year_month_sample_counts": [
            {key: _json_value(value) for key, value in row.items()}
            for row in year_month_counts
        ],
        "monthly_grid_cell_counts": monthly_grid_counts,
        "monthly_grid_cell_stability": grid_stability,
        "lag_sample_estimate": {
            "input_window": int(input_window),
            "estimated_available_samples": int(total_lag_samples),
            "rejected_nonconsecutive_targets": int(
                total_rejected_lag_targets
            ),
        },
        "storage_estimate": estimate_full_artifact_storage(
            total_converted_rows,
            total_lag_samples,
            task_count=task_count,
        ),
        "blocking_issues": list(dict.fromkeys(blocking_issues)),
        "warnings": list(dict.fromkeys(warnings)),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, default=_json_value)
    return report


def audit_processed_era5_csv(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    expected_rows: int | None = None,
    chunksize: int = 250_000,
) -> dict[str, Any]:
    """Audit a full processed CSV in bounded memory."""
    source = resolve_project_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Processed ERA5 CSV not found: {source}")
    key_columns = [
        "region",
        "year",
        "month",
        "latitude",
        "longitude",
    ]
    required = list(PROJECT_SCHEMA_COLUMNS)
    numeric_columns = [column for column in required if column != "region"]
    total_rows = 0
    regions: set[str] = set()
    period_counts: dict[tuple[str, int, int], int] = {}
    non_finite_counts = {column: 0 for column in numeric_columns}
    missing_counts = {column: 0 for column in required}
    minima = {column: float("inf") for column in numeric_columns}
    maxima = {column: float("-inf") for column in numeric_columns}
    sums = {column: 0.0 for column in numeric_columns}
    finite_counts = {column: 0 for column in numeric_columns}
    key_hashes: list[np.ndarray] = []
    columns_seen: list[str] | None = None

    for chunk in pd.read_csv(source, chunksize=chunksize):
        if columns_seen is None:
            columns_seen = list(chunk.columns)
            missing_required = [
                column for column in required if column not in chunk.columns
            ]
            if missing_required:
                raise ValueError(
                    f"Processed ERA5 CSV missing columns: {missing_required}"
                )
        total_rows += len(chunk)
        regions.update(str(value) for value in chunk["region"].dropna().unique())
        missing_counts["region"] += int(chunk["region"].isna().sum())
        counts = chunk.groupby(["region", "year", "month"]).size()
        for key, count in counts.items():
            normalized = (str(key[0]), int(key[1]), int(key[2]))
            period_counts[normalized] = (
                period_counts.get(normalized, 0) + int(count)
            )
        key_hashes.append(
            pd.util.hash_pandas_object(
                chunk[key_columns],
                index=False,
            ).to_numpy(dtype=np.uint64)
        )
        for column in numeric_columns:
            values = pd.to_numeric(chunk[column], errors="coerce")
            missing_counts[column] += int(values.isna().sum())
            finite_mask = np.isfinite(values)
            non_finite_counts[column] += int((~finite_mask).sum())
            finite = values[finite_mask]
            if len(finite):
                minima[column] = min(minima[column], float(finite.min()))
                maxima[column] = max(maxima[column], float(finite.max()))
                sums[column] += float(finite.sum())
                finite_counts[column] += len(finite)

    combined_hashes = (
        np.concatenate(key_hashes)
        if key_hashes else np.array([], dtype=np.uint64)
    )
    combined_hashes.sort()
    duplicate_key_rows = int(
        (combined_hashes[1:] == combined_hashes[:-1]).sum()
    )
    region_periods: dict[str, list[pd.Period]] = {}
    for region_name, year, month in period_counts:
        region_periods.setdefault(region_name, []).append(
            pd.Period(f"{year:04d}-{month:02d}", freq="M")
        )
    missing_months_by_region = {
        region_name: [
            str(period)
            for period in pd.period_range(
                min(periods), max(periods), freq="M"
            ).difference(pd.PeriodIndex(periods))
        ]
        for region_name, periods in region_periods.items()
    }
    monthly_counts_by_region: dict[str, list[int]] = {}
    for (region_name, _, _), count in period_counts.items():
        monthly_counts_by_region.setdefault(region_name, []).append(count)
    stability = {
        region_name: {
            "stable": len(set(counts)) == 1,
            "unique_monthly_sample_counts": sorted(set(counts)),
        }
        for region_name, counts in monthly_counts_by_region.items()
    }
    blocking: list[str] = []
    if expected_rows is not None and total_rows != expected_rows:
        blocking.append(
            f"Row count {total_rows:,} does not match preflight estimate "
            f"{expected_rows:,}"
        )
    if duplicate_key_rows:
        blocking.append(
            f"Found {duplicate_key_rows:,} duplicate "
            "region/year/month/latitude/longitude keys"
        )
    total_non_finite = sum(non_finite_counts.values())
    if total_non_finite:
        blocking.append(
            f"Found {total_non_finite:,} non-finite numeric values"
        )
    for region_name, missing_months in missing_months_by_region.items():
        if missing_months:
            blocking.append(
                f"{region_name} is missing months: {missing_months}"
            )
    unstable = [
        region_name
        for region_name, details in stability.items()
        if not details["stable"]
    ]
    if unstable:
        blocking.append(
            f"Monthly sample counts are unstable for regions: {unstable}"
        )

    report = {
        "audit_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if not blocking else "warning",
        "path": str(source.resolve()),
        "size_bytes": int(source.stat().st_size),
        "row_count": int(total_rows),
        "expected_rows": expected_rows,
        "columns": columns_seen or [],
        "regions": sorted(regions),
        "region_count": len(regions),
        "date_range_by_region": {
            region_name: {
                "start": str(min(periods)),
                "end": str(max(periods)),
                "missing_months": missing_months_by_region[region_name],
            }
            for region_name, periods in region_periods.items()
        },
        "region_year_month_sample_counts": [
            {
                "region": region_name,
                "year": year,
                "month": month,
                "sample_count": count,
            }
            for (region_name, year, month), count in sorted(
                period_counts.items()
            )
        ],
        "monthly_sample_count_stability": stability,
        "duplicate_key_rows": duplicate_key_rows,
        "duplicate_detection": (
            "64-bit pandas hash over "
            "region/year/month/latitude/longitude; collision risk is negligible"
        ),
        "missing_counts": missing_counts,
        "non_finite_counts": non_finite_counts,
        "variable_summary": {
            column: {
                "min": minima[column] if finite_counts[column] else None,
                "mean": (
                    sums[column] / finite_counts[column]
                    if finite_counts[column] else None
                ),
                "max": maxima[column] if finite_counts[column] else None,
            }
            for column in numeric_columns
        },
        "blocking_issues": blocking,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite processed audit: {destination}"
            )
        with destination.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, default=_json_value)
    return report
