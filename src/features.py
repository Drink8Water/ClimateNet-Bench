"""Create physically informed features and anomaly variables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import FEATURES_PATH, SAMPLE_DATA_PATH, ensure_directories
from physical_features import add_physical_features

ANOMALY_COLUMNS = ["temperature", "precipitation", "radiation", "soil_moisture", "evaporation"]


def load_climate_data(input_path: Path) -> pd.DataFrame:
    """Load climate data with a clear error if it is missing."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing climate data at {input_path}. "
            "Run python src/make_sample_data.py for synthetic data or "
            "python src/preprocess_era5.py for ERA5-Land data."
        )
    return pd.read_csv(input_path)


def add_climatology_and_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """Add monthly climatology and anomaly columns by region and month."""
    features = data.copy()
    group_keys = ["region", "month"]

    for column in ANOMALY_COLUMNS:
        climatology_column = f"{column}_climatology"
        anomaly_column = f"{column}_anomaly"
        features[climatology_column] = features.groupby(group_keys)[column].transform("mean")
        features[anomaly_column] = features[column] - features[climatology_column]

    return features


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create all model features."""
    features = add_physical_features(data)
    features = add_climatology_and_anomalies(features)
    return features


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create ML features from climate data CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=SAMPLE_DATA_PATH,
        help="Input climate CSV. Defaults to synthetic sample data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FEATURES_PATH,
        help="Output feature CSV. Defaults to data/processed/features.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Load sample data, create features, and save them."""
    ensure_directories()
    args = parse_args()
    data = load_climate_data(args.input)
    features = create_features(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Saved features to {args.output}")
    print(f"Rows: {len(features):,}, Columns: {len(features.columns)}")


if __name__ == "__main__":
    main()
