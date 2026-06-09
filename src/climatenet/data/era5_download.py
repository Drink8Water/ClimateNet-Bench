"""ERA5-Land download utilities using the CDS API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from climatenet.utils.paths import ensure_directory, resolve_project_path

ALL_MONTHS = [f"{month:02d}" for month in range(1, 13)]


def check_cds_credentials() -> None:
    """Fail early with a clear message if CDS API credentials are missing."""
    cdsapirc = Path.home() / ".cdsapirc"
    has_env_credentials = bool(os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"))

    if not cdsapirc.exists() and not has_env_credentials:
        raise FileNotFoundError(
            "CDS API credentials were not found.\n\n"
            "Create a CDS account, accept the ERA5-Land dataset licence, then create:\n"
            f"  {cdsapirc}\n\n"
            "with content like:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <your-personal-api-key>\n\n"
            "Alternatively set CDSAPI_URL and CDSAPI_KEY environment variables."
        )


def build_request(region_name: str, era5_config: dict[str, Any], years: list[str], months: list[str]) -> dict[str, Any]:
    """Build one CDS API request for a configured region."""
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": era5_config["variables"],
        "year": years,
        "month": months,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": era5_config["regions"][region_name]["cds_area"],
    }


def era5_output_path(raw_dir: Path, region_name: str, years: list[str], months: list[str]) -> Path:
    """Create a deterministic NetCDF output path for a region request."""
    safe_region_name = region_name.lower().replace(" ", "_")
    year_label = f"{years[0]}_{years[-1]}" if len(years) > 1 else years[0]
    month_label = "all_months" if len(months) == 12 else "_".join(months)
    return raw_dir / f"era5_land_{safe_region_name}_{year_label}_{month_label}.nc"


def download_region(
    client: Any,
    region_name: str,
    era5_config: dict[str, Any],
    years: list[str],
    months: list[str],
) -> Path:
    """Download one region to a NetCDF file."""
    raw_dir = ensure_directory(resolve_project_path(era5_config["raw_dir"]))
    output_path = era5_output_path(raw_dir, region_name, years, months)

    if output_path.exists():
        print(f"Skipping existing file: {output_path}")
        return output_path

    request = build_request(region_name, era5_config, years, months)
    print(f"Downloading {region_name} to {output_path}")
    client.retrieve(era5_config["dataset_name"], request, str(output_path))
    return output_path


def download_era5_land(
    data_config: dict[str, Any],
    full_request: bool = False,
    region: str | None = None,
) -> list[Path]:
    """Download ERA5-Land monthly data from CDS using data_config.yaml."""
    check_cds_credentials()

    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError("cdsapi is required. Install dependencies with: pip install -r requirements.txt") from exc

    era5_config = data_config["era5"]
    years = list(era5_config["full_years"] if full_request else era5_config["safe_default_years"])
    months = ALL_MONTHS if full_request else list(era5_config["safe_default_months"])
    regions = [region] if region else list(era5_config["regions"].keys())

    print(f"Dataset: {era5_config['dataset_name']}")
    print(f"Years: {years}")
    print(f"Months: {months}")
    print(f"Regions: {regions}")

    client = cdsapi.Client()
    return [download_region(client, name, era5_config, years, months) for name in regions]
