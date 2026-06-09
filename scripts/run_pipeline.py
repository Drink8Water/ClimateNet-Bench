"""Run the ClimateNet feature pipeline from YAML configuration."""

from __future__ import annotations

import argparse

from climatenet.data.loaders import load_csv, save_csv
from climatenet.features.pipeline import build_features
from climatenet.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build ClimateNet features from configured input data.")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Build and save feature table."""
    args = parse_args()
    config = load_yaml(args.data_config)
    data = load_csv(config["input_data_path"], required_columns=config.get("required_raw_columns"))
    features = build_features(data)
    save_csv(features, config["features_path"])
    print(f"Saved features to {config['features_path']}")
    print(f"Rows: {len(features):,}, Columns: {len(features.columns)}")


if __name__ == "__main__":
    main()
