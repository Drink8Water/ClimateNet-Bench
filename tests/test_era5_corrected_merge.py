"""Tests for corrected ERA5-Land accumulated-field replacement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climatenet.data.era5_corrected_merge import (
    execute_corrected_merge,
    plan_corrected_merge,
    validate_corrected_config,
    write_source_data_status,
)
from climatenet.utils.config import load_yaml


def _times(day: int = 1) -> list[pd.Timestamp]:
    return [
        pd.Timestamp(year=int(year), month=int(month), day=day)
        for year, months in {
            "2022": ["09", "10", "11", "12"],
            "2023": [f"{value:02d}" for value in range(1, 13)],
        }.items()
        for month in months
    ]


def _write_full(path: Path, *, longitude=(10.0, 11.0)) -> None:
    shape = (16, 2, 2)
    variables = {
        name: (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, index + 1.0, dtype=np.float32),
        )
        for index, name in enumerate(
            ["t2m", "tp", "ssrd", "swvl1", "u10", "v10", "e"]
        )
    }
    dataset = xr.Dataset(
        variables,
        coords={
            "valid_time": _times(day=1),
            "latitude": [1.0, 0.0],
            "longitude": list(longitude),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    dataset.close()


def _write_patch(
    path: Path,
    *,
    times: list[pd.Timestamp] | None = None,
    longitude=(10.0, 11.0),
) -> None:
    patch_times = times or _times(day=2)
    shape = (len(patch_times), 2, 2)
    dataset = xr.Dataset(
        {
            name: (
                ("valid_time", "latitude", "longitude"),
                np.full(shape, 100.0 + index, dtype=np.float32),
            )
            for index, name in enumerate(["ssrd", "tp", "e"])
        },
        coords={
            "valid_time": patch_times,
            "latitude": [1.0, 0.0],
            "longitude": list(longitude),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    dataset.close()


def _config(tmp_path: Path) -> dict:
    regions = {}
    outputs = []
    for region, safe in [("Sahara", "sahara"), ("East China", "east_china")]:
        output = tmp_path / f"{safe}_corrected.nc"
        outputs.append(str(output))
        regions[region] = {
            "old_full": str(tmp_path / f"{safe}_old.nc"),
            "patch": str(tmp_path / f"{safe}_patch.nc"),
            "output": str(output),
        }
    return {
        "corrected_merge": {
            "replaced_variables": ["ssrd", "tp", "e"],
            "replaced_periods": {
                "2022": ["09", "10", "11", "12"],
                "2023": [f"{value:02d}" for value in range(1, 13)],
            },
            "regions": regions,
        },
        "era5": {"input_files": outputs},
    }


def _prepare(config: dict) -> None:
    for values in config["corrected_merge"]["regions"].values():
        _write_full(Path(values["old_full"]))
        _write_patch(Path(values["patch"]))


def test_merge_matches_year_month_not_exact_timestamp_and_manifest(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _prepare(config)

    manifest = execute_corrected_merge(config, "Sahara")
    output = Path(manifest["output"])

    assert manifest["replacement_month_count"] == 16
    assert manifest["replacement_value_slices"] == 48
    assert manifest["old_time_coordinate"] == "valid_time"
    assert manifest["patch_time_coordinate"] == "valid_time"
    assert manifest["output_sha256"]
    assert Path(manifest["manifest_path"]).exists()
    with xr.open_dataset(output) as dataset:
        assert (
            pd.DatetimeIndex(dataset.valid_time.values).day.unique().tolist()
            == [1]
        )
        assert np.all(dataset["ssrd"].values == 100.0)
        assert np.all(dataset["tp"].values == 101.0)
        assert np.all(dataset["e"].values == 102.0)
        assert np.all(dataset["t2m"].values == 1.0)


def test_merge_rejects_grid_mismatch_and_missing_patch_month(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _prepare(config)
    patch = Path(
        config["corrected_merge"]["regions"]["Sahara"]["patch"]
    )
    _write_patch(patch, longitude=(10.0, 12.0))
    with pytest.raises(ValueError, match="grids differ"):
        plan_corrected_merge(config, "Sahara")

    _write_patch(patch, times=_times(day=2)[:-1])
    with pytest.raises(ValueError, match="Patch periods"):
        plan_corrected_merge(config, "Sahara")


def test_merge_refuses_overwrite_and_config_uses_only_corrected_inputs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _prepare(config)
    merge = validate_corrected_config(config)

    assert config["era5"]["input_files"] == [
        merge["regions"]["Sahara"]["output"],
        merge["regions"]["East China"]["output"],
    ]
    assert all(
        "old" not in path and "patch" not in path
        for path in config["era5"]["input_files"]
    )
    execute_corrected_merge(config, "Sahara")
    with pytest.raises(FileExistsError, match="overwrite"):
        execute_corrected_merge(config, "Sahara")


def test_source_data_status_writer(tmp_path: Path) -> None:
    runs = [tmp_path / "run", tmp_path / "summary"]
    for directory in runs:
        directory.mkdir()
    outputs = write_source_data_status(
        runs,
        corrected_dataset_paths=[tmp_path / "corrected.csv"],
        known_issue_url="https://example.test/known-issue",
        created_at="2026-07-30T00:00:00+00:00",
    )

    assert len(outputs) == 2
    payload = json.loads(outputs[0].read_text())
    assert payload["status"] == "source_data_invalid"
    assert payload["affected_variables"] == ["ssrd", "tp", "e"]
    assert payload["keep_for_audit"] is True


def test_production_corrected_config_never_scans_old_raw_directory() -> None:
    config = load_yaml(
        "configs/data_config_external_corrected_2019_2023.yaml"
    )
    merge = validate_corrected_config(config)

    assert config["era5"]["stream_output"] is True
    assert config["era5"]["raw_dir"].endswith("era5_land_corrected")
    assert len(config["era5"]["input_files"]) == 2
    assert config["era5"]["input_files"] == [
        merge["regions"]["Sahara"]["output"],
        merge["regions"]["East China"]["output"],
    ]
    assert all(
        "corrected_accumulated.nc" in path
        for path in config["era5"]["input_files"]
    )
