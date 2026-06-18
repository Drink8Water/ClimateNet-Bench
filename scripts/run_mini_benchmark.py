#!/usr/bin/env python
"""Run a mini-benchmark on synthetic data with three baseline models.

This script is designed for CI smoke testing and fast iterative
development.  It:

1. Creates a tiny synthetic dataset.
2. Builds an anomaly target and lag features.
3. Constructs at least one event label using hydroclimate label functions.
4. Runs ClimatologyBaseline, PersistenceBaseline, and LightGBMBaseline.
5. Evaluates each model and writes predictions, metrics, and leaderboard.

Outputs
-------
- ``outputs/mini_benchmark/predictions_{model}.csv``
- ``outputs/mini_benchmark/metrics_{model}.json``
- ``outputs/mini_benchmark/leaderboard/v1_mini.csv``

Usage
-----
.. code-block:: bash

    python scripts/run_mini_benchmark.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_mini_benchmark")


# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    n_regions: int = 3,
    n_points: int = 4,
    n_years: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a small synthetic dataset suitable for mini-benchmark testing.

    Returns a DataFrame with columns:
    sample_id, year, month, lat, lon, region, climate_zone,
    evaporation_anomaly, temperature_anomaly, precipitation_anomaly,
    soil_moisture_anomaly, radiation_anomaly,
    {target}_lag1, {target}_lag2, ..., {target}_lag6
    """
    rng = np.random.default_rng(seed)

    region_specs = [
        ("Sahara", "arid", 22.0, 10.0),
        ("East China", "monsoon", 30.0, 115.0),
        ("Amazon", "tropical_humid", -3.0, -60.0),
        ("Central Europe", "temperate", 50.0, 10.0),
        ("Western US", "semi_arid", 38.0, -120.0),
    ]

    rows = []
    for ri in range(n_regions):
        region_name, climate_zone, base_lat, base_lon = region_specs[ri]
        for pi in range(n_points):
            lat = base_lat + pi * 0.5
            lon = base_lon + pi * 0.5
            for yi in range(n_years):
                year = 2020 + yi
                for month in range(1, 13):
                    # Deterministic seasonal signal + noise
                    temp_base = 30 if region_name == "Sahara" else 20
                    temp = temp_base + 12 * np.cos(2 * np.pi * (month - 7) / 12) + rng.normal(0, 2)
                    precip_base = 5 if region_name == "Sahara" else 60
                    precip = max(0, precip_base + 50 * np.cos(2 * np.pi * (month - 7) / 12) + rng.normal(0, 8))
                    rad = 250 + 50 * np.cos(2 * np.pi * (month - 6) / 12) + rng.normal(0, 10)
                    sm = max(0.02, 0.15 + 0.002 * precip + rng.normal(0, 0.03))
                    evap = max(0, 0.015 * rad + 0.08 * temp + 12 * sm + rng.normal(0, 1.5))

                    rows.append(
                        {
                            "sample_id": f"{region_name}_{pi}_{year}_{month:02d}",
                            "year": year,
                            "month": month,
                            "lat": round(lat, 4),
                            "lon": round(lon, 4),
                            "region": region_name,
                            "climate_zone": climate_zone,
                            "temperature": round(temp, 3),
                            "precipitation": round(precip, 3),
                            "radiation": round(rad, 3),
                            "soil_moisture": round(sm, 4),
                            "evaporation": round(evap, 3),
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------


def _add_features(df: pd.DataFrame, target_col: str = "evaporation_anomaly") -> pd.DataFrame:
    """Add anomaly columns, lag features, and event labels."""
    data = df.copy()

    # --- Anomalies (by region × month) ---
    for col in ["temperature", "precipitation", "radiation", "soil_moisture", "evaporation"]:
        clim = data.groupby(["region", "month"])[col].transform("mean")
        data[f"{col}_anomaly"] = data[col] - clim

    # --- Lag features ---
    target_col_clean = target_col.replace("_anomaly", "")
    for lag in range(1, 7):
        data[f"{target_col}_lag{lag}"] = data.groupby(["region", "lat", "lon"])[
            target_col
        ].shift(lag)

    # Drop rows with NaN lags (first 6 months per grid cell)
    data = data.dropna(subset=[f"{target_col}_lag{lag}" for lag in range(1, 7)]).reset_index(
        drop=True
    )

    return data


# ---------------------------------------------------------------------------
# 3. Split
# ---------------------------------------------------------------------------


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simple temporal holdout: train=2020-2022, val=2023, test=2024."""
    train = df[df["year"].isin([2020, 2021, 2022])].copy()
    val = df[df["year"] == 2023].copy()
    test = df[df["year"] == 2024].copy()

    if train.empty:
        raise ValueError("Train set is empty. Check year ranges.")
    if val.empty:
        # Use part of train as val if no val year exists
        val = train.sample(frac=0.1, random_state=42)
    if test.empty:
        # Use the last year as test
        last_year = df["year"].max()
        test = df[df["year"] == last_year].copy()
        train = df[df["year"] < last_year].copy()

    return train, val, test


# ---------------------------------------------------------------------------
# 4. Event labels
# ---------------------------------------------------------------------------


def _build_event_thresholds(train_df: pd.DataFrame) -> dict:
    """Fit event thresholds on training data only.

    Returns the thresholds dict used by ``evaluate_model_on_split`` to
    construct both true and predicted event labels.
    """
    from climatenet.evaluation.hydroclimate_labels import fit_event_thresholds

    return fit_event_thresholds(train_df)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the mini-benchmark pipeline."""
    output_root = _PROJECT_ROOT / "outputs" / "mini_benchmark"
    lb_dir = output_root / "leaderboard"
    lb_path = lb_dir / "v1_mini.csv"

    # Clean start
    import shutil

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lb_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Mini Benchmark ===")

    # --- 1. Create synthetic data ---
    logger.info("Step 1: Creating synthetic dataset …")
    raw = _make_synthetic_data(n_regions=3, n_points=4, n_years=5, seed=42)
    target_col = "evaporation_anomaly"
    df = _add_features(raw, target_col=target_col)
    logger.info("  Dataset: %d samples, %d columns", len(df), len(df.columns))

    # --- 2. Build split ---
    logger.info("Step 2: Building temporal split …")
    train_df, val_df, test_df = _temporal_split(df)
    logger.info(
        "  Train: %d, Val: %d, Test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # --- 3. Event thresholds (fitted on train only) ---
    logger.info("Step 3: Fitting event thresholds on train data …")
    event_thresholds = _build_event_thresholds(train_df)
    # Only evaluate event types derivable from the target variable (evaporation_anomaly).
    #   evaporation_deficit → defined on evaporation_anomaly ✓
    #   soil_moisture_drought → defined on soil_moisture_anomaly ✗
    #   compound_hot_dry → requires temperature + soil moisture ✗
    event_cols = ["evaporation_deficit"]

    # Feature columns for LightGBM
    feature_cols = [
        f"{target_col}_lag1",
        f"{target_col}_lag2",
        f"{target_col}_lag3",
        f"{target_col}_lag4",
        f"{target_col}_lag5",
        f"{target_col}_lag6",
        "temperature_anomaly",
        "precipitation_anomaly",
        "soil_moisture_anomaly",
        "radiation_anomaly",
    ]

    # Filter to columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info("  Feature columns: %s", feature_cols)
    logger.info("  Event types: %s", event_cols)

    # --- 4. Import evaluation modules ---
    from climatenet.evaluation.leaderboard import update_leaderboard
    from climatenet.evaluation.runner import evaluate_model_on_split
    from climatenet.models.baselines import (
        ClimatologyBaseline,
        LightGBMBaseline,
        PersistenceBaseline,
    )

    models_to_run: list[tuple[str, Any]] = [
        ("climatology", ClimatologyBaseline(target_col=target_col, predict_zero=True)),
        (
            "persistence",
            PersistenceBaseline(target_col=target_col, lag_col=f"{target_col}_lag1"),
        ),
    ]

    # LightGBM is optional
    try:
        lgb = LightGBMBaseline(
            n_estimators=50,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )
        models_to_run.append(("lightgbm", lgb))
    except ImportError:
        logger.warning("LightGBM not installed — skipping lightgbm baseline.")

    # --- 5. Run each model ---
    for model_name, model in models_to_run:
        logger.info("Step 4.%s: Running %s …", model_name[0] if model_name else "", model_name)

        result = evaluate_model_on_split(
            model=model,
            train_df=train_df,
            val_df=val_df if model_name == "lightgbm" else None,
            test_df=test_df,
            target_col=target_col,
            feature_cols=feature_cols if model_name != "climatology" else None,
            event_cols=event_cols,
            event_thresholds=event_thresholds,
        )

        preds_df = result["predictions_df"]
        metrics = result["metrics_overall"]

        # Save predictions
        pred_path = output_root / f"predictions_{model_name}.csv"
        preds_df.to_csv(pred_path, index=False)
        logger.info("  Predictions saved to %s", pred_path)

        # Save metrics JSON
        metrics_path = output_root / f"metrics_{model_name}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("  Metrics saved to %s", metrics_path)

        # Update leaderboard
        update_leaderboard(
            metrics=metrics,
            model_name=model_name,
            split_name="temporal_holdout",
            output_path=lb_path,
        )
        logger.info("  Leaderboard updated: %s", lb_path)

        # Print key metrics
        logger.info(
            "  %s: RMSE=%.4f, MAE=%.4f, R²=%.4f",
            model_name,
            metrics.get("rmse", float("nan")),
            metrics.get("mae", float("nan")),
            metrics.get("r2", float("nan")),
        )

    # --- 6. Print leaderboard ---
    logger.info("=== Leaderboard ===")
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        print()
        print(lb.to_string(index=False))
        print()
    else:
        logger.warning("Leaderboard not found at %s", lb_path)

    logger.info("=== Mini Benchmark Complete ===")


if __name__ == "__main__":
    main()
