"""ERA5-Land NetCDF preprocessing into ClimateNet tabular schema."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

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


def open_era5_subset(
    path: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    bbox: dict[str, list[float]] | None = None,
    max_grid_cells: int | None = None,
) -> Any:
    """Open, validate and eagerly load a bounded ERA5-Land subset.

    ``bbox`` uses ``{"latitude": [min, max], "longitude": [min, max]}``.
    The explicit grid-cell limit is a dry-run guardrail: it raises instead of
    silently sampling or accidentally materialising a full regional request.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required. Install dependencies with: pip install -r requirements.txt") from exc

    dataset = xr.open_dataset(path)
    missing_variables = [name for name in ERA5_RENAME_MAP if name not in dataset.data_vars]
    if missing_variables:
        available = list(dataset.data_vars)
        dataset.close()
        raise ValueError(
            f"{path} is missing expected ERA5 variables {missing_variables}. "
            f"Available variables: {available}"
        )

    time_name = next(
        (name for name in ["valid_time", "time"] if name in dataset.coords),
        None,
    )
    if time_name is None:
        available = list(dataset.coords)
        dataset.close()
        raise ValueError(
            f"{path} has no 'valid_time' or 'time' coordinate. "
            f"Available coordinates: {available}"
        )
    subset = dataset[list(ERA5_RENAME_MAP)]
    if start is not None or end is not None:
        subset = subset.sel(
            {time_name: slice(start or None, end or None)}
        )
    if bbox is not None:
        for coordinate in ["latitude", "longitude"]:
            if coordinate not in subset.coords:
                dataset.close()
                raise ValueError(
                    f"{path} is missing required coordinate {coordinate!r}"
                )
            bounds = bbox.get(coordinate)
            if bounds is None or len(bounds) != 2:
                dataset.close()
                raise ValueError(
                    f"bbox.{coordinate} must contain [min, max]"
                )
            lower, upper = sorted(float(value) for value in bounds)
            subset = subset.where(
                (subset[coordinate] >= lower)
                & (subset[coordinate] <= upper),
                drop=True,
            )

    grid_cells = int(
        subset.sizes.get("latitude", 0)
        * subset.sizes.get("longitude", 0)
    )
    if grid_cells == 0 or subset.sizes.get(time_name, 0) == 0:
        dataset.close()
        raise ValueError(
            f"ERA5 subset is empty for path={path}, start={start}, "
            f"end={end}, bbox={bbox}"
        )
    if max_grid_cells is not None and grid_cells > max_grid_cells:
        dataset.close()
        raise ValueError(
            f"ERA5 subset contains {grid_cells:,} grid cells, exceeding "
            f"max_grid_cells={max_grid_cells:,}. Narrow the bounding box."
        )
    subset = subset.load()
    dataset.close()
    return subset


def preprocess_era5_file(
    path: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    bbox: dict[str, list[float]] | None = None,
    max_grid_cells: int | None = None,
    region: str | None = None,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Read one ERA5-Land NetCDF file and return ClimateNet tabular columns.

    Optional subset arguments are intended for readiness audits and dry-runs.
    Existing callers retain the original full-file behaviour.
    """
    resolved_region = region or infer_region_from_filename(path)
    dataset = open_era5_subset(
        path,
        start=start,
        end=end,
        bbox=bbox,
        max_grid_cells=max_grid_cells,
    )
    dataset = dataset.rename(ERA5_RENAME_MAP)
    data = dataset.to_dataframe().reset_index()
    dataset.close()

    time_column = find_time_column(data)
    data[time_column] = pd.to_datetime(data[time_column])
    data["region"] = resolved_region
    data["year"] = data[time_column].dt.year
    data["month"] = data[time_column].dt.month

    data = convert_units(data[PROJECT_SCHEMA_COLUMNS])
    if drop_invalid:
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


def preprocess_era5_files_streaming(
    input_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Write explicit NetCDF inputs incrementally with overwrite protection."""
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing processed data: {output_path}"
        )
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if partial_path.exists():
        raise FileExistsError(
            f"Partial processed file already exists: {partial_path}. "
            "Inspect it before retrying; it will not be overwritten."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    regions: set[str] = set()
    years: set[int] = set()
    columns: list[str] = []
    try:
        for index, path in enumerate(input_paths):
            print(f"Preprocessing {path}")
            frame = preprocess_era5_file(path)
            columns = list(frame.columns)
            regions.update(str(value) for value in frame["region"].unique())
            years.update(int(value) for value in frame["year"].unique())
            frame.to_csv(
                partial_path,
                mode="w" if index == 0 else "a",
                header=index == 0,
                index=False,
            )
            total_rows += len(frame)
            del frame
            gc.collect()
        partial_path.replace(output_path)
    except Exception:
        # Deliberately preserve the uniquely named partial for diagnosis.
        raise
    return {
        "output_path": str(output_path),
        "rows": total_rows,
        "columns": columns,
        "regions": sorted(regions),
        "years": sorted(years),
        "size_bytes": int(output_path.stat().st_size),
    }


def preprocess_era5_from_config(
    data_config: dict,
) -> pd.DataFrame | dict[str, Any]:
    """Preprocess ERA5 data using paths from data_config.yaml."""
    era5_config = data_config["era5"]
    input_dir = ensure_directory(resolve_project_path(era5_config["raw_dir"]))
    output_path = resolve_project_path(era5_config["processed_path"])
    configured_files = era5_config.get("input_files")
    if configured_files:
        input_paths = [resolve_project_path(path) for path in configured_files]
        missing = [str(path) for path in input_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Configured ERA5 input files do not exist: {missing}"
            )
        if len(set(input_paths)) != len(input_paths):
            raise ValueError("era5.input_files contains duplicate paths")
        if bool(era5_config.get("stream_output", False)):
            return preprocess_era5_files_streaming(input_paths, output_path)
        frames = [preprocess_era5_file(path) for path in input_paths]
        climate_data = pd.concat(frames, ignore_index=True)
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing processed data: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        climate_data.to_csv(output_path, index=False)
        return climate_data
    return preprocess_era5_directory(input_dir, output_path)
