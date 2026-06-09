"""Download ERA5-Land data using the ClimateNet package."""

from __future__ import annotations

import argparse
import sys

from climatenet.data.era5_download import download_era5_land
from climatenet.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download ERA5-Land monthly NetCDF files.")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--full-request", action="store_true")
    parser.add_argument("--region", choices=["Sahara", "East China"])
    return parser.parse_args()


def main() -> None:
    """Download ERA5-Land data."""
    args = parse_args()
    data_config = load_yaml(args.data_config)
    try:
        paths = download_era5_land(data_config, full_request=args.full_request, region=args.region)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    print("Downloaded files:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
