"""Preprocess ERA5-Land NetCDF files using the ClimateNet package."""

from __future__ import annotations

import argparse

from climatenet.data.era5_preprocess import preprocess_era5_from_config
from climatenet.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess ERA5-Land NetCDF files into CSV.")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Preprocess ERA5-Land data."""
    args = parse_args()
    data_config = load_yaml(args.data_config)
    climate_data = preprocess_era5_from_config(data_config)
    print(f"Saved ERA5 tabular data to {data_config['era5']['processed_path']}")
    print(f"Rows: {len(climate_data):,}, Columns: {len(climate_data.columns)}")
    print("Regions:", ", ".join(sorted(climate_data["region"].unique())))


if __name__ == "__main__":
    main()
