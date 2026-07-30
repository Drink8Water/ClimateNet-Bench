"""Safe retrieval and audit helpers for the ERA5-Land accumulated patch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.data.era5_download import check_cds_credentials

PATCH_DATASET = "reanalysis-era5-land-monthly-means"
PATCH_PRODUCT_TYPE = "monthly_averaged_reanalysis_by_hour_of_day"
PATCH_TIME = "00:00"
PATCH_VARIABLES = {
    "surface_solar_radiation_downwards",
    "total_precipitation",
    "total_evaporation",
}
PATCH_PERIODS = {
    "2022": ["09", "10", "11", "12"],
    "2023": [f"{month:02d}" for month in range(1, 13)],
}
SHORT_NAMES = {
    "surface_solar_radiation_downwards": "ssrd",
    "total_precipitation": "tp",
    "total_evaporation": "e",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_patch_config(config: dict[str, Any]) -> dict[str, Any]:
    patch = config.get("era5_patch")
    if not isinstance(patch, dict):
        raise ValueError("Config must contain era5_patch")
    if patch.get("dataset_name") != PATCH_DATASET:
        raise ValueError(f"Patch dataset must be {PATCH_DATASET}")
    if patch.get("product_type") != PATCH_PRODUCT_TYPE:
        raise ValueError(f"Patch product_type must be {PATCH_PRODUCT_TYPE}")
    if patch.get("time") != PATCH_TIME:
        raise ValueError("Patch time must be exactly 00:00")
    variables = set(patch.get("variables", []))
    if variables != PATCH_VARIABLES:
        raise ValueError(
            "Patch variables must be exactly the three accumulated variables"
        )
    periods = {
        str(year): [str(month) for month in months]
        for year, months in patch.get("periods", {}).items()
    }
    if periods != PATCH_PERIODS:
        raise ValueError(
            "Patch period must be exactly 2022-09..12 and 2023-01..12"
        )
    if set(patch.get("regions", {})) != {"Sahara", "East China"}:
        raise ValueError("Patch regions must be Sahara and East China")
    return patch


def patch_output_path(patch: dict[str, Any], region: str) -> Path:
    if region not in patch["regions"]:
        raise ValueError(f"Unsupported patch region: {region}")
    safe_region = region.lower().replace(" ", "_")
    return Path(patch["raw_dir"]) / (
        f"era5_land_{safe_region}_202209_202312_accumulated_patch_"
        "hourly_monthly_00.nc"
    )


def build_patch_requests(
    config: dict[str, Any], region: str
) -> list[dict[str, Any]]:
    patch = validate_patch_config(config)
    if region not in patch["regions"]:
        raise ValueError(f"Unsupported patch region: {region}")
    common = {
        "product_type": [PATCH_PRODUCT_TYPE],
        "variable": list(patch["variables"]),
        "time": [PATCH_TIME],
        "data_format": patch.get("data_format", "netcdf"),
        "download_format": patch.get("download_format", "unarchived"),
        "area": patch["regions"][region]["cds_area"],
    }
    return [
        {
            **common,
            "year": [year],
            "month": months,
        }
        for year, months in PATCH_PERIODS.items()
    ]


def request_manifest(
    config: dict[str, Any], region: str
) -> dict[str, Any]:
    patch = validate_patch_config(config)
    output = patch_output_path(patch, region)
    requests = build_patch_requests(config, region)
    return {
        "dataset_name": PATCH_DATASET,
        "region": region,
        "output_path": str(output),
        "request_count": len(requests),
        "requests": requests,
        "expected_periods": [
            f"{year}-{month}"
            for year, months in PATCH_PERIODS.items()
            for month in months
        ],
        "overwrite_policy": "skip_nonempty_refuse_zero_byte",
        "merge_scope": (
            "two patch request segments only; never merged with old full data"
        ),
    }


def save_request_manifest(
    config: dict[str, Any], region: str
) -> Path:
    patch = validate_patch_config(config)
    output = patch_output_path(patch, region)
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = output.with_suffix(".request.json")
    destination.write_text(
        json.dumps(request_manifest(config, region), indent=2),
        encoding="utf-8",
    )
    return destination


def _time_name(dataset: Any) -> str:
    name = next(
        (candidate for candidate in ["valid_time", "time"] if candidate in dataset.coords),
        None,
    )
    if name is None:
        raise ValueError("Downloaded patch segment has no time coordinate")
    return name


def _validate_segment(path: Path, expected_year: str) -> None:
    import xarray as xr

    dataset = xr.open_dataset(path)
    try:
        variables = set(dataset.data_vars)
        missing = set(SHORT_NAMES.values()) - variables
        if missing:
            raise ValueError(f"{path} missing patch variables: {sorted(missing)}")
        time_name = _time_name(dataset)
        periods = {
            str(value)
            for value in pd.PeriodIndex(dataset[time_name].values, freq="M")
        }
        expected = {
            f"{expected_year}-{month}" for month in PATCH_PERIODS[expected_year]
        }
        if periods != expected:
            raise ValueError(
                f"{path} periods {sorted(periods)} != {sorted(expected)}"
            )
    finally:
        dataset.close()


def merge_patch_segments(
    segment_paths: list[Path], output_path: Path
) -> Path:
    """Merge the two safe patch segments, never the old full dataset."""
    import xarray as xr

    if output_path.exists():
        if output_path.stat().st_size > 0:
            return output_path
        raise FileExistsError(f"Refusing zero-byte output: {output_path}")
    partial = output_path.with_suffix(".nc.partial")
    if partial.exists():
        raise FileExistsError(f"Inspect existing partial before retry: {partial}")
    datasets = [xr.open_dataset(path) for path in segment_paths]
    try:
        time_names = [_time_name(dataset) for dataset in datasets]
        if len(set(time_names)) != 1:
            raise ValueError(f"Patch segment time coordinates differ: {time_names}")
        for coordinate in ["latitude", "longitude"]:
            reference = datasets[0][coordinate].values
            if any(
                not np.array_equal(reference, dataset[coordinate].values)
                for dataset in datasets[1:]
            ):
                raise ValueError(f"Patch segment {coordinate} grids differ")
        time_name = time_names[0]
        merged = xr.concat(datasets, dim=time_name).sortby(time_name).load()
        periods = [
            str(value)
            for value in pd.PeriodIndex(merged[time_name].values, freq="M")
        ]
        expected = [
            f"{year}-{month}"
            for year, months in PATCH_PERIODS.items()
            for month in months
        ]
        if periods != expected:
            raise ValueError(
                f"Merged patch periods {periods} != expected {expected}"
            )
        merged.attrs["ClimateNet_patch_product_type"] = PATCH_PRODUCT_TYPE
        merged.attrs["ClimateNet_patch_time"] = PATCH_TIME
        merged.attrs["ClimateNet_patch_scope"] = "2022-09 through 2023-12"
        merged.to_netcdf(partial)
        merged.close()
        partial.replace(output_path)
    finally:
        for dataset in datasets:
            dataset.close()
    return output_path


def download_patch_region(
    config: dict[str, Any],
    region: str,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    patch = validate_patch_config(config)
    output = patch_output_path(patch, region)
    manifest_path = save_request_manifest(config, region)
    if output.exists():
        if output.stat().st_size > 0:
            return {
                "status": "skipped_existing",
                "output_path": str(output),
                "size_bytes": output.stat().st_size,
                "sha256": _sha256(output),
                "request_manifest": str(manifest_path),
            }
        raise FileExistsError(f"Refusing to overwrite zero-byte file: {output}")
    if client is None:
        check_cds_credentials()
        try:
            import cdsapi
        except ImportError as exc:
            raise ImportError("cdsapi is required for --execute") from exc
        client = cdsapi.Client()
    requests = build_patch_requests(config, region)
    segment_paths: list[Path] = []
    for request in requests:
        year = request["year"][0]
        segment = output.with_name(f"{output.stem}.segment_{year}.nc")
        segment_paths.append(segment)
        if segment.exists():
            if segment.stat().st_size == 0:
                raise FileExistsError(
                    f"Refusing zero-byte segment; inspect it: {segment}"
                )
        else:
            client.retrieve(PATCH_DATASET, request, str(segment))
        _validate_segment(segment, year)
    merge_patch_segments(segment_paths, output)
    return {
        "status": "downloaded",
        "output_path": str(output),
        "size_bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "request_manifest": str(manifest_path),
        "segments": [str(path) for path in segment_paths],
    }
