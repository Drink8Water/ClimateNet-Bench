#!/usr/bin/env python
"""Build the ClimateNet-Bench forecasting dataset from processed features.

Usage
-----
.. code-block:: bash

    # Default: load features.csv, build 6-month windows, save to data/processed/
    python scripts/build_forecasting_dataset.py

    # Custom paths
    python scripts/build_forecasting_dataset.py \\
        --input data/processed/features.csv \\
        --output-dir data/processed \\
        --prefix forecasting

    # Smoke test with synthetic data
    python scripts/build_forecasting_dataset.py --synthetic

Outputs
-------
- ``{output_dir}/{prefix}_samples.csv`` — one row per forecasting sample
- ``{output_dir}/{prefix}_metadata.json`` — dataset-level metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path (for src/config.py compatibility).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.data.forecasting_dataset import build_and_save
from climatenet.data.loaders import load_csv
from climatenet.utils.paths import resolve_project_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_forecasting_dataset")


# ---------------------------------------------------------------------------
# synthetic fallback
# ---------------------------------------------------------------------------


def _build_synthetic_features() -> "pd.DataFrame":
    """Generate synthetic demo features for smoke testing.

    The generated data is explicitly labelled as synthetic demo data and
    must NOT be reported as benchmark results.
    """
    import numpy as np
    import pandas as pd

    from climatenet.utils.random import set_random_seed

    set_random_seed(42)

    # Use the same synthetic generator as the original make_sample_data.py
    # but go directly to a feature table so we don't need the old src/config.py.
    regions_config = {
        "Sahara": {"lat_range": (15.0, 30.0), "lon_range": (-20.0, 30.0)},
        "East China": {"lat_range": (20.0, 35.0), "lon_range": (105.0, 122.0)},
    }

    rng = np.random.default_rng(42)
    rows = []
    years = range(2010, 2021)
    grid_points_per_region = 5  # small for smoke test

    for region_name, bounds in regions_config.items():
        lats = rng.uniform(*bounds["lat_range"], size=grid_points_per_region)
        lons = rng.uniform(*bounds["lon_range"], size=grid_points_per_region)
        for lat, lon in zip(lats, lons):
            for year in years:
                for month in range(1, 13):
                    # deterministic seasonal signal + noise
                    base_temp = 30 if region_name == "Sahara" else 17
                    temp = base_temp + 10 * np.cos(2 * np.pi * (month - 7) / 12) + rng.normal(0, 1.5)
                    precip_base = 5 if region_name == "Sahara" else 55
                    precip = max(0, precip_base + 40 * np.cos(2 * np.pi * (month - 7) / 12) + rng.normal(0, 10))
                    rad = 250 + 40 * np.cos(2 * np.pi * (month - 6) / 12) + rng.normal(0, 10)
                    sm = max(0.02, 0.15 + 0.001 * precip + rng.normal(0, 0.02))
                    wind = np.sqrt(rng.normal(2, 1) ** 2 + rng.normal(1, 1) ** 2)
                    evap = max(0, 0.015 * rad + 0.08 * temp + 12 * sm + 0.2 * wind + rng.normal(0, 1))

                    rows.append(
                        {
                            "region": region_name,
                            "year": year,
                            "month": month,
                            "latitude": round(float(lat), 4),
                            "longitude": round(float(lon), 4),
                            "temperature": round(float(temp), 3),
                            "precipitation": round(float(precip), 3),
                            "radiation": round(float(rad), 3),
                            "soil_moisture": round(float(sm), 4),
                            "u_wind": round(float(rng.normal(2, 1)), 3),
                            "v_wind": round(float(rng.normal(1, 1)), 3),
                            "evaporation": round(float(evap), 3),
                        }
                    )

    raw = pd.DataFrame(rows)

    # --- feature engineering (inline, avoids importing the full pipeline) ---
    features = raw.copy()
    features["wind_speed"] = np.sqrt(features["u_wind"] ** 2 + features["v_wind"] ** 2)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    features["dryness_proxy"] = features["radiation"] / (features["precipitation"] + 1e-6)
    features["saturation_vapor_pressure"] = 0.6108 * np.exp(
        (17.27 * features["temperature"]) / (features["temperature"] + 237.3)
    )

    # anomalies
    for col in ["temperature", "precipitation", "radiation", "soil_moisture", "evaporation"]:
        clim = features.groupby(["region", "month"])[col].transform("mean")
        features[f"{col}_anomaly"] = features[col] - clim

    logger.warning(
        "*** SYNTHETIC DEMO DATA — NOT REAL CLIMATE DATA. "
        "Do NOT report as benchmark results. ***"
    )
    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ClimateNet-Bench forecasting dataset"
    )
    parser.add_argument(
        "--input",
        default="data/processed/features.csv",
        help="Path to the processed feature CSV (default: data/processed/features.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for output files (default: data/processed)",
    )
    parser.add_argument(
        "--prefix",
        default="forecasting",
        help="Prefix for output filenames (default: forecasting)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=6,
        help="Number of past months per sample (default: 6)",
    )
    parser.add_argument(
        "--target",
        default="evaporation_anomaly",
        help="Target column name (default: evaporation_anomaly)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic demo data instead of loading from disk",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # load or generate features
    # ------------------------------------------------------------------
    if args.synthetic:
        logger.info("Building synthetic demo features …")
        features = _build_synthetic_features()
    else:
        input_path = resolve_project_path(args.input)
        logger.info("Loading features from %s", input_path)
        features = load_csv(input_path)

    logger.info("Feature table: %d rows × %d columns", len(features), len(features.columns))
    logger.info("Regions: %s", sorted(features["region"].unique().tolist()))
    logger.info("Years: %s", sorted(features["year"].unique().tolist()))

    # ------------------------------------------------------------------
    # build forecasting dataset
    # ------------------------------------------------------------------
    output_dir = resolve_project_path(args.output_dir)
    logger.info("Building forecasting samples (sequence_length=%d) …", args.sequence_length)

    samples, metadata, errors = build_and_save(
        data=features,
        output_dir=output_dir,
        prefix=args.prefix,
        target_column=args.target,
        sequence_length=args.sequence_length,
    )

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Forecasting Dataset Built Successfully")
    print("=" * 60)
    print(f"  Samples:       {metadata.total_samples:,}")
    print(f"  Grid cells:    {metadata.grid_cells:,}")
    print(f"  Regions:       {metadata.regions}")
    print(f"  Climate types: {metadata.climate_types}")
    print(f"  Sequence len:  {metadata.sequence_length}")
    print(f"  Dropped rows:  {metadata.dropped_incomplete:,}")
    print(f"  Output dir:    {output_dir}")
    print()

    if errors:
        print("  Validation messages:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  Validation:    all checks passed ✓")
    print("=" * 60)

    # Print a few sample rows so the user can verify lag alignment
    print()
    print("--- Sample rows for manual inspection ---")
    lag_cols = [f"temperature_anomaly_lag_{i}" for i in range(1, args.sequence_length + 1)]
    display_cols = [
        "sample_id",
        "target_year",
        "target_month",
        "input_window_start",
        "input_window_end",
        "y_true",
        *lag_cols,
    ]
    print(samples[display_cols].head(3).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
