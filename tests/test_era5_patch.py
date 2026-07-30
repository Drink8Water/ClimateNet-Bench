"""Tests for guarded ERA5-Land accumulated patch retrieval and audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climatenet.data.era5_patch import (
    PATCH_PERIODS,
    build_patch_requests,
    download_patch_region,
    patch_output_path,
    save_request_manifest,
    validate_patch_config,
)
from climatenet.data.era5_patch_audit import audit_patch_files


def _config(tmp_path: Path) -> dict:
    return {
        "era5_patch": {
            "dataset_name": "reanalysis-era5-land-monthly-means",
            "product_type": (
                "monthly_averaged_reanalysis_by_hour_of_day"
            ),
            "time": "00:00",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "raw_dir": str(tmp_path / "patch"),
            "audit_json": str(tmp_path / "audit.json"),
            "audit_markdown": str(tmp_path / "audit.md"),
            "periods": {
                year: list(months) for year, months in PATCH_PERIODS.items()
            },
            "variables": [
                "surface_solar_radiation_downwards",
                "total_precipitation",
                "total_evaporation",
            ],
            "regions": {
                "Sahara": {
                    "cds_area": [30, -20, 15, 30],
                    "old_full_file": str(tmp_path / "sahara_old.nc"),
                },
                "East China": {
                    "cds_area": [35, 105, 20, 122],
                    "old_full_file": str(tmp_path / "east_old.nc"),
                },
            },
        }
    }


def _write_dataset(
    path: Path,
    times: list[pd.Timestamp],
    *,
    value: float,
    longitude: tuple[float, float] = (10.0, 11.0),
    patch: bool = False,
) -> None:
    shape = (len(times), 2, 2)
    dataset = xr.Dataset(
        {
            "ssrd": (
                ("valid_time", "latitude", "longitude"),
                np.full(shape, value, dtype=np.float32),
                {"units": "J m**-2"},
            ),
            "tp": (
                ("valid_time", "latitude", "longitude"),
                np.full(shape, value / 1e10, dtype=np.float32),
                {"units": "m"},
            ),
            "e": (
                ("valid_time", "latitude", "longitude"),
                np.full(shape, -value / 1e10, dtype=np.float32),
                {"units": "m"},
            ),
        },
        coords={
            "valid_time": times,
            "latitude": [1.0, 0.0],
            "longitude": list(longitude),
        },
        attrs={"history": "stream mnth"},
    )
    if patch:
        dataset.attrs["ClimateNet_patch_product_type"] = (
            "monthly_averaged_reanalysis_by_hour_of_day"
        )
        dataset.attrs["ClimateNet_patch_time"] = "00:00"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    dataset.close()


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, dataset: str, request: dict, target: str) -> None:
        self.calls.append(
            {"dataset": dataset, "request": request, "target": target}
        )
        year = int(request["year"][0])
        times = [
            pd.Timestamp(year=year, month=int(month), day=1)
            for month in request["month"]
        ]
        _write_dataset(Path(target), times, value=20_000_000.0)


def test_patch_request_is_exactly_scoped_and_manifest_saved(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    requests = build_patch_requests(config, "Sahara")

    assert len(requests) == 2
    assert requests[0]["year"] == ["2022"]
    assert requests[0]["month"] == ["09", "10", "11", "12"]
    assert requests[1]["year"] == ["2023"]
    assert requests[1]["month"] == [f"{month:02d}" for month in range(1, 13)]
    assert requests[0]["time"] == ["00:00"]
    assert set(requests[0]["variable"]) == {
        "surface_solar_radiation_downwards",
        "total_precipitation",
        "total_evaporation",
    }

    manifest = save_request_manifest(config, "Sahara")
    payload = json.loads(manifest.read_text())
    assert payload["request_count"] == 2
    assert payload["expected_periods"][0] == "2022-09"
    assert payload["expected_periods"][-1] == "2023-12"


def test_patch_config_rejects_extra_variable_or_time(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["era5_patch"]["variables"].append("2m_temperature")
    with pytest.raises(ValueError, match="exactly"):
        validate_patch_config(config)

    config = _config(tmp_path)
    config["era5_patch"]["periods"]["2022"] = ["08", "09"]
    with pytest.raises(ValueError, match="period"):
        validate_patch_config(config)


def test_download_merges_segments_and_skips_existing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = _FakeClient()

    first = download_patch_region(config, "Sahara", client=client)
    second = download_patch_region(config, "Sahara", client=client)

    assert first["status"] == "downloaded"
    assert second["status"] == "skipped_existing"
    assert len(client.calls) == 2
    output = Path(first["output_path"])
    assert output.stat().st_size > 0
    with xr.open_dataset(output) as dataset:
        assert dataset.sizes["valid_time"] == 16
        assert set(dataset.data_vars) == {"ssrd", "tp", "e"}


def test_patch_audit_validates_variables_time_and_grid(tmp_path: Path) -> None:
    config = _config(tmp_path)
    old_times = list(pd.date_range("2019-01-01", "2023-12-01", freq="MS"))
    patch_times = list(pd.date_range("2022-09-01", "2023-12-01", freq="MS"))
    for region, old_name in [
        ("Sahara", "sahara_old.nc"),
        ("East China", "east_old.nc"),
    ]:
        _write_dataset(tmp_path / old_name, old_times, value=10_000_000.0)
        output = patch_output_path(config["era5_patch"], region)
        _write_dataset(
            output,
            patch_times,
            value=20_000_000.0,
            patch=True,
        )

    report = audit_patch_files(config)
    assert report["status"] == "ready"
    assert all(item["grid_aligned"] for item in report["files"])
    assert report["comparison_summary"]["patch_identical_to_old_ratio"] == 0

    east = patch_output_path(config["era5_patch"], "East China")
    _write_dataset(
        east,
        patch_times,
        value=20_000_000.0,
        longitude=(10.0, 12.0),
        patch=True,
    )
    failed = audit_patch_files(config)
    assert failed["status"] == "failed"
    assert any("grid" in issue for issue in failed["blocking_issues"])
