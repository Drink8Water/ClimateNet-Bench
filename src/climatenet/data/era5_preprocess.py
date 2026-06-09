"""ERA5-Land NetCDF preprocessing into ClimateNet tabular schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from climatenet.utils.paths import ensure_directory, resolve_project_path

ERA5_RENAME_MAP = {
    "t2m": "temperature",
    "tp": "precipitation",
    "ssrd": "radiation",
    "swvl1": "soil_moisture",
    "u10": "u_wind",
    "v10": "v_wind",
    "e": "evaporation",
}

PROJECT_SCHEMA_COLUMNS = [
    "region",
    "year",
    "month",
    "latitude",
    "longitude",
    "temperature",
    "precipitation",
    "radiation",
    "soil_moisture",
    "u_wind",
    "v_wind",
    "evaporation",
]


def find_time_column(data: pd.DataFrame) -> str:
    """Find the ERA5 time coordinate after converting xarray to dataframe."""
    for candidate in ["valid_time", "time"]:
        if candidate in data.columns:
            return candidate
    raise ValueError("Could not find a time coordinate column named 'valid_time' or 'time'.")


def infer_region_from_filename(path: Path) -> str:
    """Infer region name from filenames created by the ERA5 downloader."""
    filename = path.name.lower()
    if "sahara" in filename:
        return "Sahara"
    if "east_china" in filename:
        return "East China"
    raise ValueError(
        f"Could not infer region from filename: {path.name}. "
        "Use filenames containing 'sahara' or 'east_china'."
    )


def convert_units(data: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5-Land units to ClimateNet's beginner-friendly units.

    - t2m: K -> degrees Celsius.
    - tp: m/day water equivalent -> monthly total mm.
    - ssrd: J m-2/day -> monthly total MJ m-2.
    - e: m/day water equivalent -> positive monthly total mm.
    - swvl1: m3 m-3, kept unchanged.
    - u10/v10: m s-1, kept unchanged.
    """
    converted = data.copy()
    days_in_month = pd.to_datetime(
        {"year": converted["year"], "month": converted["month"], "day": 1}
    ).dt.days_in_month

    converted["temperature"] = converted["temperature"] - 273.15
    converted["precipitation"] = converted["precipitation"] * 1000.0 * days_in_month
    converted["radiation"] = converted["radiation"] / 1_000_000.0 * days_in_month
    converted["evaporation"] = -converted["evaporation"] * 1000.0 * days_in_month
    return converted


def preprocess_era5_file(path: Path) -> pd.DataFrame:
    """Read one ERA5-Land NetCDF file and return ClimateNet tabular columns."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required. Install dependencies with: pip install -r requirements.txt") from exc

    region = infer_region_from_filename(path)
    dataset = xr.open_dataset(path)

    missing_variables = [name for name in ERA5_RENAME_MAP if name not in dataset.data_vars]
    if missing_variables:
        available = list(dataset.data_vars)
        raise ValueError(
            f"{path} is missing expected ERA5 variables {missing_variables}. "
            f"Available variables: {available}"
        )

    dataset = dataset[list(ERA5_RENAME_MAP)].rename(ERA5_RENAME_MAP)
    data = dataset.to_dataframe().reset_index()
    dataset.close()

    time_column = find_time_column(data)
    data[time_column] = pd.to_datetime(data[time_column])
    data["region"] = region
    data["year"] = data[time_column].dt.year
    data["month"] = data[time_column].dt.month

    data = convert_units(data[PROJECT_SCHEMA_COLUMNS])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    numeric_columns = [column for column in PROJECT_SCHEMA_COLUMNS if column != "region"]
    data[numeric_columns] = data[numeric_columns].round(6)
    return data


def preprocess_era5_directory(input_dir: Path, output_path: Path) -> pd.DataFrame:
    """Preprocess every NetCDF file in a directory and save one CSV."""
    netcdf_paths = sorted(input_dir.glob("*.nc"))
    if not netcdf_paths:
        raise FileNotFoundError(f"No NetCDF files found in {input_dir}. Run the ERA5 download step first.")

    frames = []
    for path in netcdf_paths:
        print(f"Preprocessing {path}")
        frames.append(preprocess_era5_file(path))

    climate_data = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    climate_data.to_csv(output_path, index=False)
    return climate_data


def preprocess_era5_from_config(data_config: dict) -> pd.DataFrame:
    """Preprocess ERA5 data using paths from data_config.yaml."""
    era5_config = data_config["era5"]
    input_dir = ensure_directory(resolve_project_path(era5_config["raw_dir"]))
    output_path = resolve_project_path(era5_config["processed_path"])
    return preprocess_era5_directory(input_dir, output_path)
