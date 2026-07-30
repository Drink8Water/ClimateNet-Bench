"""Tests for guarded ERA5-Land readiness audit and dry-run entry point."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climatenet.data.era5_audit import (
    audit_era5_files,
    audit_processed_era5_csv,
)
from climatenet.data.era5_dry_run import (
    run_era5_dry_run,
    validate_era5_dry_run_config,
)
from climatenet.data.era5_preprocess import preprocess_era5_from_config
from climatenet.features.physical import (
    add_physical_features,
    build_physical_features_csv,
)
from climatenet.utils.config import load_yaml


def _write_era5_netcdf(
    path: Path,
    times: list[str],
    *,
    include_all_variables: bool = True,
    inject_non_finite: bool = False,
) -> None:
    shape = (len(times), 2, 2)
    variables = {
        "t2m": (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, 300.0),
            {"units": "K"},
        ),
        "tp": (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, 0.001),
            {"units": "m"},
        ),
        "ssrd": (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, 10_000_000.0),
            {"units": "J m**-2"},
        ),
        "e": (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, -0.002),
            {"units": "m of water equivalent"},
        ),
        "swvl1": (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, 0.2),
            {"units": "m**3 m**-3"},
        ),
        "u10": (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, 2.0),
            {"units": "m s**-1"},
        ),
        "v10": (
            ("valid_time", "latitude", "longitude"),
            np.full(shape, 1.0),
            {"units": "m s**-1"},
        ),
    }
    if not include_all_variables:
        variables.pop("e")
    if inject_non_finite:
        variables["t2m"][1][0, 0, 0] = np.nan
        variables["tp"][1][0, 0, 1] = np.inf
    dataset = xr.Dataset(
        data_vars=variables,
        coords={
            "valid_time": pd.to_datetime(times),
            "latitude": [20.1, 20.0],
            "longitude": [0.0, 5.0],
        },
    )
    dataset.to_netcdf(path)


def test_audit_detects_nonfinite_values_and_missing_month_warning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny.nc"
    output = tmp_path / "audit.json"
    _write_era5_netcdf(
        path,
        ["2020-01-01", "2020-03-01"],
        inject_non_finite=True,
    )

    report = audit_era5_files(
        [path],
        region="Sahara",
        max_grid_cells=10,
        max_total_bytes=10_000_000,
        input_window=1,
        output_path=output,
    )

    assert report["status"] == "warning"
    assert report["monthly_coverage"]["continuous"] is False
    assert report["monthly_coverage"]["missing_months"] == ["2020-02"]
    assert any("non-continuous" in warning for warning in report["warnings"])
    assert report["input_files"][0]["variables"]["t2m"]["raw_summary"][
        "non_finite_count"
    ] == 1
    assert report["converted_variable_summary_before_row_filter"][
        "precipitation"
    ][
        "positive_inf_count"
    ] == 1
    assert report["row_filter_summary"]["partially_invalid_rows"] == 2
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "warning"


def test_audit_missing_required_variable_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.nc"
    _write_era5_netcdf(
        path,
        ["2020-01-01", "2020-02-01"],
        include_all_variables=False,
    )

    with pytest.raises(ValueError, match="missing expected ERA5 variables.*e"):
        audit_era5_files(
            [path],
            region="Sahara",
            max_grid_cells=10,
            max_total_bytes=10_000_000,
        )


def test_dry_run_config_guardrails_block_full_experiment() -> None:
    config = load_yaml("configs/benchmark/era5_land_dry_run.yaml")
    validate_era5_dry_run_config(config)

    assert config["dry_run"] is True
    assert config["synthetic"] is False
    assert len(config["split_protocols"]) == 2
    assert "temporal_holdout" not in config["split_protocols"]
    assert all(
        (model if isinstance(model, str) else model["name"]) != "tcn"
        for model in config["models"]
    )

    unsafe = copy.deepcopy(config)
    unsafe["dry_run"] = False
    with pytest.raises(ValueError, match="dry_run: true"):
        validate_era5_dry_run_config(unsafe)

    unsafe = copy.deepcopy(config)
    unsafe["models"].append({"name": "tcn"})
    with pytest.raises(ValueError, match="TCN"):
        validate_era5_dry_run_config(unsafe)


def test_formal_era5_config_cannot_fall_back_to_synthetic_features() -> None:
    formal = load_yaml("configs/benchmark/evap_anomaly_v1.yaml")
    smoke = load_yaml("configs/benchmark/smoke_test.yaml")

    assert formal["synthetic"] is False
    assert formal["features_path"] == (
        "data/processed/era5_physical_features.csv"
    )
    assert formal["features_path"] != smoke.get(
        "features_path", "data/processed/features.csv"
    )


def test_external_full_config_is_explicit_and_does_not_include_dry_run_file() -> None:
    config = load_yaml("configs/data_config_external_full.yaml")
    era5 = config["era5"]

    assert era5["full_years"] == ["2019", "2020", "2021", "2022", "2023"]
    assert set(era5["regions"]) == {"Sahara", "East China"}
    assert era5["stream_output"] is True
    assert len(era5["input_files"]) == 2
    assert all("2019_2023_all_months.nc" in path for path in era5["input_files"])
    assert all("2019_2021" not in path for path in era5["input_files"])
    assert era5["processed_path"].startswith(
        "/media/drink8water/拯救者PSSD/ClimateNet-Bench/data/processed/"
    )


def test_real_dry_run_uses_an_isolated_run_directory(tmp_path: Path) -> None:
    path = tmp_path / "tiny.nc"
    months = pd.date_range("2019-01-01", periods=12, freq="MS")
    _write_era5_netcdf(path, [date.isoformat() for date in months])
    config = {
        "benchmark_name": "tiny-era5-dry-run",
        "dataset_name": "test ERA5",
        "dataset_version": "test",
        "data_source": "ERA5-Land-dry-run",
        "synthetic": False,
        "dry_run": True,
        "target": "evaporation_anomaly",
        "target_column": "y_true",
        "input_window": 6,
        "random_seed": 7,
        "real_data": {
            "netcdf_paths": [str(path)],
            "region": "Sahara",
            "start": "2019-01",
            "end": "2019-12",
            "bbox": {
                "latitude": [20.0, 20.1],
                "longitude": [0.0, 5.0],
            },
            "max_grid_cells": 10,
            "max_input_bytes": 10_000_000,
        },
        "preprocessing": {
            "train_only": True,
            "standardize_features": True,
        },
        "split_protocols": ["random_split"],
        "feature_sets": {
            "base": {
                "features": [
                    "latitude",
                    "longitude",
                    "month_sin",
                    "month_cos",
                ]
            }
        },
        "models": [],
        "metrics": {"primary": ["rmse"]},
    }
    output_root = tmp_path / "benchmark_runs"

    registry = run_era5_dry_run(config, output_root=output_root)
    run_dir = registry.path.parent
    metadata = json.loads(
        (run_dir / "run_metadata.json").read_text(encoding="utf-8")
    )

    assert run_dir.parent == output_root
    assert run_dir != output_root
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "data_audit" / "era5_readiness.json").exists()
    assert (
        run_dir
        / "data_audit"
        / "era5_dry_run_physical_features.csv"
    ).exists()
    assert metadata["real_data_audit"]["status"] == "ready"
    assert metadata["real_data_audit"]["prepared_features_sha256"]
    assert metadata["real_data_audit"]["source_files"][0]["sha256"]


def test_streaming_full_preprocess_uses_only_explicit_files_and_wont_overwrite(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    months = pd.date_range("2019-01-01", periods=12, freq="MS")
    sahara = raw_dir / "era5_land_sahara_2019_2023_all_months.nc"
    east_china = raw_dir / "era5_land_east_china_2019_2023_all_months.nc"
    old_dry_run = raw_dir / "era5_land_sahara_2019_2021_all_months.nc"
    for path in [sahara, east_china, old_dry_run]:
        _write_era5_netcdf(path, [date.isoformat() for date in months])
    output = tmp_path / "processed" / "full.csv"
    config = {
        "era5": {
            "raw_dir": str(raw_dir),
            "processed_path": str(output),
            "input_files": [str(sahara), str(east_china)],
            "stream_output": True,
        }
    }

    summary = preprocess_era5_from_config(config)

    assert isinstance(summary, dict)
    assert summary["rows"] == 96
    assert summary["regions"] == ["East China", "Sahara"]
    assert len(pd.read_csv(output)) == 96
    assert not output.with_suffix(".csv.partial").exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        preprocess_era5_from_config(config)


def test_processed_csv_audit_detects_duplicate_keys(tmp_path: Path) -> None:
    csv_path = tmp_path / "processed.csv"
    rows = pd.DataFrame(
        {
            "region": ["Sahara", "Sahara"],
            "year": [2020, 2020],
            "month": [1, 1],
            "latitude": [20.0, 20.0],
            "longitude": [10.0, 10.0],
            "temperature": [25.0, 25.0],
            "precipitation": [1.0, 1.0],
            "radiation": [500.0, 500.0],
            "soil_moisture": [0.1, 0.1],
            "u_wind": [2.0, 2.0],
            "v_wind": [1.0, 1.0],
            "evaporation": [3.0, 3.0],
        }
    )
    rows.to_csv(csv_path, index=False)

    report = audit_processed_era5_csv(csv_path, chunksize=1)

    assert report["status"] == "warning"
    assert report["duplicate_key_rows"] == 1
    assert any("duplicate" in issue for issue in report["blocking_issues"])


def test_formal_physical_features_log_extreme_dryness_without_inf() -> None:
    raw = pd.DataFrame(
        {
            "u_wind": [1.0],
            "v_wind": [2.0],
            "month": [7],
            "radiation": [900.0],
            "precipitation": [0.0],
            "temperature": [42.0],
            "evaporation": [-0.1],
        }
    )

    result = add_physical_features(raw)

    assert result.loc[0, "dryness_proxy"] == pytest.approx(900_000_000.0)
    assert np.isfinite(result.loc[0, "dryness_proxy_log1p"])
    assert result.loc[0, "evaporation"] == -0.1


def test_streamed_physical_features_refuse_overwrite_and_write_audit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.csv"
    output = tmp_path / "physical.csv"
    audit = tmp_path / "audit.json"
    pd.DataFrame(
        {
            "u_wind": [1.0, 2.0],
            "v_wind": [2.0, 3.0],
            "month": [1, 2],
            "radiation": [500.0, 600.0],
            "precipitation": [0.0, 10.0],
            "temperature": [20.0, 25.0],
            "evaporation": [-0.1, 2.0],
        }
    ).to_csv(source, index=False)

    report = build_physical_features_csv(
        source, output, audit_path=audit, chunksize=1
    )

    assert report["row_count"] == 2
    assert report["negative_evaporation_count"] == 1
    assert report["non_finite_counts"]["dryness_proxy_log1p"] == 0
    assert report["sha256"]
    assert audit.exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_physical_features_csv(source, output)
