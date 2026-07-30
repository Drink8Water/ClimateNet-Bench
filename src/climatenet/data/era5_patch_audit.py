"""Structural and value audit for corrected ERA5-Land patch files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climatenet.data.era5_patch import (
    PATCH_PERIODS,
    PATCH_PRODUCT_TYPE,
    PATCH_TIME,
    SHORT_NAMES,
    patch_output_path,
    validate_patch_config,
)
from climatenet.data.radiation_consistency import sha256_file


def _time_name(dataset: xr.Dataset) -> str:
    name = next(
        (candidate for candidate in ["valid_time", "time"] if candidate in dataset.coords),
        None,
    )
    if name is None:
        raise ValueError("Dataset has no time/valid_time coordinate")
    return name


def _converted(
    short_name: str, values: np.ndarray, days: np.ndarray
) -> np.ndarray:
    scale = days.reshape((-1, 1, 1))
    if short_name == "ssrd":
        return values * scale / 1_000_000.0
    if short_name == "tp":
        return values * scale * 1000.0
    if short_name == "e":
        return -values * scale * 1000.0
    raise ValueError(f"Unsupported patch variable: {short_name}")


def audit_patch_files(config: dict[str, Any]) -> dict[str, Any]:
    patch = validate_patch_config(config)
    expected_periods = [
        f"{year}-{month}"
        for year, months in PATCH_PERIODS.items()
        for month in months
    ]
    files: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []
    blocking: list[str] = []
    for region, region_config in patch["regions"].items():
        patch_path = patch_output_path(patch, region)
        old_path = Path(region_config["old_full_file"])
        if not patch_path.is_file() or not old_path.is_file():
            raise FileNotFoundError(
                f"Missing patch/old input for {region}: {patch_path}, {old_path}"
            )
        corrected = xr.open_dataset(patch_path)
        old = xr.open_dataset(old_path)
        try:
            corrected_time = _time_name(corrected)
            old_time = _time_name(old)
            times = pd.DatetimeIndex(corrected[corrected_time].values)
            periods = [str(value) for value in pd.PeriodIndex(times, freq="M")]
            data_variables = set(corrected.data_vars)
            expected_variables = set(SHORT_NAMES.values())
            if data_variables != expected_variables:
                blocking.append(
                    f"{region} patch variables {sorted(data_variables)} != "
                    f"{sorted(expected_variables)}"
                )
            if periods != expected_periods:
                blocking.append(
                    f"{region} patch periods {periods} != {expected_periods}"
                )
            if any(timestamp.hour != 0 for timestamp in times):
                blocking.append(f"{region} patch contains non-00:00 timestamps")
            grid_aligned = all(
                np.array_equal(corrected[name].values, old[name].values)
                for name in ["latitude", "longitude"]
            )
            if not grid_aligned:
                blocking.append(f"{region} patch grid does not align with old file")
            history = str(corrected.attrs.get("history", ""))
            product = corrected.attrs.get(
                "ClimateNet_patch_product_type"
            )
            if product != PATCH_PRODUCT_TYPE:
                blocking.append(
                    f"{region} missing expected embedded patch product_type"
                )
            if "mnth" not in history:
                warnings.append(
                    f"{region} NetCDF history does not explicitly contain mnth"
                )

            old_times = pd.DatetimeIndex(old[old_time].values)
            old_periods = pd.PeriodIndex(old_times, freq="M")
            old_lookup = {str(period): index for index, period in enumerate(old_periods)}
            baseline_years = old_times.year <= 2021
            baseline_months = old_times.month
            variable_reports = {}
            for short_name in sorted(expected_variables):
                attrs = dict(corrected[short_name].attrs)
                corrected_values = np.asarray(
                    corrected[short_name].values, dtype=np.float64
                )
                old_values = np.asarray(old[short_name].values, dtype=np.float64)
                days = times.days_in_month.to_numpy(dtype=np.float64)
                corrected_converted = _converted(
                    short_name, corrected_values, days
                )
                nan_counts = np.isnan(corrected_values).sum(axis=(1, 2))
                positive_inf = np.isposinf(corrected_values).sum(axis=(1, 2))
                negative_inf = np.isneginf(corrected_values).sum(axis=(1, 2))
                variable_reports[short_name] = {
                    "attrs": attrs,
                    "shape": list(corrected_values.shape),
                    "nan_count_unique": sorted(
                        int(value) for value in np.unique(nan_counts)
                    ),
                    "positive_inf_count": int(positive_inf.sum()),
                    "negative_inf_count": int(negative_inf.sum()),
                    "mask_stable": bool(len(np.unique(nan_counts)) == 1),
                }
                if positive_inf.sum() or negative_inf.sum():
                    blocking.append(f"{region} {short_name} contains Inf")
                for index, period in enumerate(periods):
                    old_index = old_lookup[period]
                    month = times[index].month
                    baseline = old_values[
                        baseline_years & (baseline_months == month)
                    ]
                    baseline_days = old_times[
                        baseline_years & (baseline_months == month)
                    ].days_in_month.to_numpy(dtype=np.float64)
                    baseline_converted = _converted(
                        short_name, baseline, baseline_days
                    )
                    old_converted = _converted(
                        short_name,
                        old_values[old_index : old_index + 1],
                        np.array([times[index].days_in_month]),
                    )
                    patch_mean = float(np.nanmean(corrected_converted[index]))
                    old_mean = float(np.nanmean(old_converted[0]))
                    baseline_mean = float(np.nanmean(baseline_converted))
                    comparisons.append(
                        {
                            "region": region,
                            "period": period,
                            "year": int(times[index].year),
                            "month": int(month),
                            "variable": short_name,
                            "baseline_2019_2021_mean": baseline_mean,
                            "old_bad_mean": old_mean,
                            "patch_mean": patch_mean,
                            "old_minus_baseline": old_mean - baseline_mean,
                            "patch_minus_baseline": patch_mean - baseline_mean,
                            "old_relative_to_baseline": (
                                (old_mean - baseline_mean) / abs(baseline_mean)
                                if baseline_mean != 0
                                else np.nan
                            ),
                            "patch_relative_to_baseline": (
                                (patch_mean - baseline_mean) / abs(baseline_mean)
                                if baseline_mean != 0
                                else np.nan
                            ),
                            "patch_minus_old": patch_mean - old_mean,
                            "patch_to_old_ratio": (
                                patch_mean / old_mean if old_mean != 0 else np.nan
                            ),
                        }
                    )
            files.append(
                {
                    "region": region,
                    "patch_path": str(patch_path),
                    "patch_size_bytes": patch_path.stat().st_size,
                    "patch_sha256": sha256_file(patch_path),
                    "old_path": str(old_path),
                    "old_size_bytes": old_path.stat().st_size,
                    "old_sha256": sha256_file(old_path),
                    "time_coordinate": corrected_time,
                    "time_start": times.min().isoformat(),
                    "time_end": times.max().isoformat(),
                    "period_count": len(times),
                    "all_timestamps_00": all(
                        timestamp.hour == 0 for timestamp in times
                    ),
                    "grid_aligned": grid_aligned,
                    "latitude_count": int(corrected.sizes["latitude"]),
                    "longitude_count": int(corrected.sizes["longitude"]),
                    "data_variables": sorted(data_variables),
                    "global_attrs": dict(corrected.attrs),
                    "variables": variable_reports,
                }
            )
        finally:
            corrected.close()
            old.close()

    comparison_frame = pd.DataFrame(comparisons)
    same_as_old = np.isclose(
        comparison_frame["patch_mean"],
        comparison_frame["old_bad_mean"],
        rtol=1e-7,
        atol=1e-9,
    )
    if bool(same_as_old.all()):
        blocking.append("Patch values are identical to known-bad old values")
    ssrd = comparison_frame[comparison_frame["variable"] == "ssrd"]
    improvement = (
        ssrd["patch_relative_to_baseline"].abs()
        < ssrd["old_relative_to_baseline"].abs()
    )
    report = {
        "status": "ready" if not blocking else "failed",
        "dataset_name": patch["dataset_name"],
        "product_type": patch["product_type"],
        "time": patch["time"],
        "expected_periods": expected_periods,
        "files": files,
        "warnings": warnings,
        "blocking_issues": blocking,
        "comparison_summary": {
            "rows": len(comparison_frame),
            "patch_identical_to_old_ratio": float(same_as_old.mean()),
            "ssrd_months_closer_to_2019_2021_baseline_ratio": float(
                improvement.mean()
            ),
        },
        "monthly_comparisons": comparisons,
    }
    return report


def write_patch_audit(
    config: dict[str, Any], report: dict[str, Any]
) -> tuple[Path, Path, Path]:
    patch = validate_patch_config(config)
    json_path = Path(patch["audit_json"])
    markdown_path = Path(patch["audit_markdown"])
    csv_path = json_path.with_name(
        "era5_accumulated_patch_202209_202312_old_vs_patch.csv"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(report["monthly_comparisons"]).to_csv(csv_path, index=False)
    files = pd.DataFrame(
        [
            {
                "region": item["region"],
                "size_bytes": item["patch_size_bytes"],
                "sha256": item["patch_sha256"],
                "period_count": item["period_count"],
                "grid_aligned": item["grid_aligned"],
            }
            for item in report["files"]
        ]
    )
    comparison = pd.DataFrame(report["monthly_comparisons"])
    yearly = (
        comparison.groupby(["region", "year", "variable"], observed=True)[
            [
                "baseline_2019_2021_mean",
                "old_bad_mean",
                "patch_mean",
            ]
        ]
        .mean()
        .reset_index()
    )
    text = f"""# ERA5-Land corrected accumulated patch audit

Status: **{report["status"]}**

Dataset: `{report["dataset_name"]}`

Product type: `{report["product_type"]}`, time `{report["time"]}`

## Files

```csv
{files.to_csv(index=False).strip()}
```

## Old versus patch yearly means

```csv
{yearly.to_csv(index=False).strip()}
```

Patch identical-to-old monthly ratio:
`{report["comparison_summary"]["patch_identical_to_old_ratio"]:.3f}`.

SSRD months closer to the 2019–2021 same-calendar-month baseline:
`{report["comparison_summary"]["ssrd_months_closer_to_2019_2021_baseline_ratio"]:.3f}`.

Blocking issues: `{report["blocking_issues"]}`

Warnings: `{report["warnings"]}`

This audit does not merge the patch into the old full NetCDF and does not
generate processed data.
"""
    markdown_path.write_text(text, encoding="utf-8")
    return json_path, markdown_path, csv_path
