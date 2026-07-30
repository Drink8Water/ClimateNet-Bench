"""Create corrected ERA5-Land files by replacing audited monthly fields."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climatenet.data.era5_patch import PATCH_PERIODS
from climatenet.data.radiation_consistency import sha256_file

REPLACED_VARIABLES = ["ssrd", "tp", "e"]
REQUIRED_FULL_VARIABLES = {"t2m", "tp", "ssrd", "swvl1", "u10", "v10", "e"}


def _time_name(dataset: xr.Dataset) -> str:
    name = next(
        (candidate for candidate in ["valid_time", "time"] if candidate in dataset.coords),
        None,
    )
    if name is None:
        raise ValueError("NetCDF has no time/valid_time coordinate")
    return name


def validate_corrected_config(config: dict[str, Any]) -> dict[str, Any]:
    merge = config.get("corrected_merge")
    if not isinstance(merge, dict):
        raise ValueError("Config must contain corrected_merge")
    if merge.get("replaced_variables") != REPLACED_VARIABLES:
        raise ValueError(f"replaced_variables must be {REPLACED_VARIABLES}")
    periods = {
        str(year): [str(month) for month in months]
        for year, months in merge.get("replaced_periods", {}).items()
    }
    if periods != PATCH_PERIODS:
        raise ValueError("Corrected merge must replace exactly 2022-09..2023-12")
    if set(merge.get("regions", {})) != {"Sahara", "East China"}:
        raise ValueError("Corrected merge requires Sahara and East China")
    expected_inputs = [
        merge["regions"][region]["output"]
        for region in ["Sahara", "East China"]
    ]
    if config.get("era5", {}).get("input_files") != expected_inputs:
        raise ValueError(
            "era5.input_files must explicitly contain only corrected outputs"
        )
    return merge


def _expected_periods() -> list[str]:
    return [
        f"{year}-{month}"
        for year, months in PATCH_PERIODS.items()
        for month in months
    ]


def plan_corrected_merge(
    config: dict[str, Any], region: str
) -> dict[str, Any]:
    merge = validate_corrected_config(config)
    if region not in merge["regions"]:
        raise ValueError(f"Unsupported corrected region: {region}")
    paths = merge["regions"][region]
    old_path = Path(paths["old_full"])
    patch_path = Path(paths["patch"])
    output_path = Path(paths["output"])
    if not old_path.is_file() or not patch_path.is_file():
        raise FileNotFoundError(
            f"Missing old/patch input: {old_path}, {patch_path}"
        )
    old = xr.open_dataset(old_path)
    patch = xr.open_dataset(patch_path)
    try:
        old_time = _time_name(old)
        patch_time = _time_name(patch)
        old_periods = [
            str(value)
            for value in pd.PeriodIndex(old[old_time].values, freq="M")
        ]
        patch_periods = [
            str(value)
            for value in pd.PeriodIndex(patch[patch_time].values, freq="M")
        ]
        if len(old_periods) != len(set(old_periods)):
            raise ValueError("Old full file has duplicate year/month entries")
        if patch_periods != _expected_periods():
            raise ValueError(
                f"Patch periods {patch_periods} != {_expected_periods()}"
            )
        missing_old = sorted(set(patch_periods) - set(old_periods))
        if missing_old:
            raise ValueError(f"Old full file lacks patch months: {missing_old}")
        if set(old.data_vars) != REQUIRED_FULL_VARIABLES:
            raise ValueError(
                f"Old variables {sorted(old.data_vars)} != "
                f"{sorted(REQUIRED_FULL_VARIABLES)}"
            )
        if set(patch.data_vars) != set(REPLACED_VARIABLES):
            raise ValueError(
                f"Patch variables {sorted(patch.data_vars)} != "
                f"{REPLACED_VARIABLES}"
            )
        grid_aligned = all(
            np.array_equal(old[name].values, patch[name].values)
            for name in ["latitude", "longitude"]
        )
        if not grid_aligned:
            raise ValueError("Old full and patch latitude/longitude grids differ")
        return {
            "region": region,
            "old_full": str(old_path),
            "patch": str(patch_path),
            "output": str(output_path),
            "output_exists": output_path.exists(),
            "output_size_bytes": (
                output_path.stat().st_size if output_path.exists() else 0
            ),
            "old_time_coordinate": old_time,
            "patch_time_coordinate": patch_time,
            "old_time_start": old_periods[0],
            "old_time_end": old_periods[-1],
            "old_month_count": len(old_periods),
            "patch_time_start": patch_periods[0],
            "patch_time_end": patch_periods[-1],
            "replacement_months": patch_periods,
            "replacement_month_count": len(patch_periods),
            "replacement_variables": REPLACED_VARIABLES,
            "replacement_value_slices": (
                len(patch_periods) * len(REPLACED_VARIABLES)
            ),
            "grid_aligned": grid_aligned,
            "grid_shape": [
                int(old.sizes["latitude"]),
                int(old.sizes["longitude"]),
            ],
            "time_handling": (
                "match patch to old by calendar year/month; preserve old full "
                "time coordinate (patch day-2 timestamp is not joined exactly)"
            ),
            "overwrite_policy": "refuse_existing_output_or_partial",
        }
    finally:
        old.close()
        patch.close()


def execute_corrected_merge(
    config: dict[str, Any], region: str
) -> dict[str, Any]:
    plan = plan_corrected_merge(config, region)
    old_path = Path(plan["old_full"])
    patch_path = Path(plan["patch"])
    output_path = Path(plan["output"])
    manifest_path = output_path.with_suffix(".merge_manifest.json")
    partial_path = output_path.with_suffix(".nc.partial")
    for candidate in [output_path, partial_path, manifest_path]:
        if candidate.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing corrected artifact: {candidate}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(old_path) as old_source, xr.open_dataset(
        patch_path
    ) as patch_source:
        old_time = _time_name(old_source)
        patch_time = _time_name(patch_source)
        old = old_source.load()
        patch = patch_source[REPLACED_VARIABLES].load()
        old_lookup = {
            str(period): index
            for index, period in enumerate(
                pd.PeriodIndex(old[old_time].values, freq="M")
            )
        }
        patch_periods = [
            str(value)
            for value in pd.PeriodIndex(patch[patch_time].values, freq="M")
        ]
        for patch_index, period in enumerate(patch_periods):
            old_index = old_lookup[period]
            for variable in REPLACED_VARIABLES:
                old[variable].values[old_index, :, :] = (
                    patch[variable].values[patch_index, :, :]
                )
        old.attrs.update(
            {
                "ClimateNet_corrected_dataset": "true",
                "ClimateNet_corrected_accumulated_period": (
                    "2022-09 through 2023-12"
                ),
                "ClimateNet_corrected_variables": ",".join(
                    REPLACED_VARIABLES
                ),
                "ClimateNet_patch_product_type": (
                    "monthly_averaged_reanalysis_by_hour_of_day"
                ),
                "ClimateNet_time_handling": plan["time_handling"],
            }
        )
        old.to_netcdf(partial_path)
        old.close()
        patch.close()
    partial_path.replace(output_path)
    manifest = {
        **plan,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "old_full_sha256": sha256_file(old_path),
        "patch_sha256": sha256_file(patch_path),
        "output_sha256": sha256_file(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def write_source_data_status(
    directories: list[str | Path],
    *,
    corrected_dataset_paths: list[str | Path],
    known_issue_url: str,
    created_at: str | None = None,
) -> list[Path]:
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    payload = {
        "status": "source_data_invalid",
        "reason": (
            "ERA5-Land monthly averaged reanalysis accumulated variables "
            "affected by ECMWF known issue, 2022-09 to 2024-02"
        ),
        "affected_variables": ["ssrd", "tp", "e"],
        "affected_benchmark_data_through": "2023-12",
        "corrected_dataset_status": "prepared_not_yet_benchmarked",
        "corrected_dataset_paths": [
            str(Path(path)) for path in corrected_dataset_paths
        ],
        "known_issue_url": known_issue_url,
        "created_at": timestamp,
        "keep_for_audit": True,
    }
    outputs = []
    for value in directories:
        directory = Path(value)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Benchmark run/summary directory not found: {directory}"
            )
        destination = directory / "source_data_status.json"
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        outputs.append(destination)
    return outputs
