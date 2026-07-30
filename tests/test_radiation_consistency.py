"""Tests for the bounded ERA5-Land radiation consistency audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climatenet.data.radiation_consistency import (
    compare_monthly_radiation,
    missing_months,
    radiation_conversions,
    summarize_grid_stability,
)
from scripts.audit_radiation_consistency import _raw_audit


def test_radiation_conversion_matches_processed_monthly_total() -> None:
    raw = np.array([10_000_000.0, 20_000_000.0])
    days = np.array([31, 28])

    converted = radiation_conversions(raw, days)

    assert converted["current_monthly_total_mj_m2"].tolist() == [
        310.0,
        560.0,
    ]
    assert converted["no_day_multiplier_mj_m2"].tolist() == [10.0, 20.0]
    assert converted["attrs_informed_daily_mean_w_m2"][0] == pytest.approx(
        115.7407407
    )

    raw_monthly = pd.DataFrame(
        {
            "region": ["Sahara", "Sahara"],
            "year": [2020, 2020],
            "month": [1, 2],
            "current_mean_mj_m2": [310.0, 560.0],
        }
    )
    processed = pd.DataFrame(
        {
            "region": ["Sahara", "Sahara"],
            "year": [2020, 2020],
            "month": [1, 2],
            "radiation_mean": [310.0, 560.0],
        }
    )
    comparison = compare_monthly_radiation(raw_monthly, processed)
    assert comparison["within_tolerance"].all()


def test_grid_stability_detects_missing_cell_and_duplicate() -> None:
    frame = pd.DataFrame(
        {
            "region": ["Sahara"] * 6,
            "year": [2020] * 4 + [2020] * 2,
            "month": [1] * 4 + [2] * 2,
            "latitude": [1, 1, 2, 2, 1, 1],
            "longitude": [10, 11, 10, 11, 10, 10],
        }
    )

    summary, warnings = summarize_grid_stability(frame)
    february = summary[summary["month"] == 2].iloc[0]

    assert february["duplicate_grid_cells"] == 1
    assert february["missing_vs_reference"] == 3
    assert warnings


def test_missing_month_warning_and_synthetic_raw_netcdf(tmp_path: Path) -> None:
    times = list(pd.date_range("2019-01-01", "2021-12-01", freq="MS"))
    times += list(pd.date_range("2023-01-01", "2023-12-01", freq="MS"))
    values = np.full((len(times), 2, 2), 10_000_000.0, dtype=np.float32)
    values[-12:] = 5_000_000.0
    dataset = xr.Dataset(
        {
            "ssrd": (
                ("valid_time", "latitude", "longitude"),
                values,
                {"units": "J m**-2", "long_name": "solar radiation"},
            )
        },
        coords={
            "valid_time": times,
            "latitude": [1.0, 0.0],
            "longitude": [10.0, 11.0],
        },
    )
    path = tmp_path / "era5_land_sahara_2019_2023_all_months.nc"
    dataset.to_netcdf(path)
    dataset.close()

    monthly, yearly, metadata, _, spatial, largest = _raw_audit(
        {"raw_files": {"Sahara": str(path)}}
    )

    assert missing_months(times) == [
        f"2022-{month:02d}" for month in range(1, 13)
    ]
    assert metadata["files"][0]["missing_months"]
    assert len(monthly) == 48
    means = yearly.set_index("year")["raw_mean_j_m2_per_day"]
    assert means.loc[2023] == pytest.approx(means.loc[2021] / 2)
    assert (spatial["fraction_decreased"] == 1.0).all()
    assert len(largest) == 4
