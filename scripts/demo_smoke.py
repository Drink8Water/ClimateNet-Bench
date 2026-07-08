"""Run a small ERA5-Land-style sample benchmark.

This script is intentionally self-contained: it creates a tiny synthetic
hydroclimate dataset, runs a fixed benchmark suite, and writes artifacts that
can be shown in the dashboard or README screenshots.

The sample data mimics the schema of the real ERA5-Land benchmark, but it is
not a scientific result.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from climatenet.evaluation.detection import (
    compute_csi,
    compute_far,
    compute_intensity_bias,
    compute_pod,
)
from climatenet.evaluation.metrics import evaluate_regression
from climatenet.models.baselines import (
    ClimatologyBaseline,
    LightGBMBaseline,
    PersistenceBaseline,
)


TARGET_COL = "evaporation_anomaly"
EVENT_COL = "soil_moisture_drought"
PRED_EVENT_COL = f"{EVENT_COL}_pred"
FEATURE_COLUMNS = [
    "temperature_anomaly",
    "precipitation_anomaly",
    "radiation_anomaly",
    "soil_moisture_anomaly",
    "dryness_proxy",
    "month_sin",
    "month_cos",
    "latitude",
    "longitude",
    f"{TARGET_COL}_lag1",
]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    train_mask: pd.Series
    test_mask: pd.Series


def build_sample_dataset(seed: int = 42) -> pd.DataFrame:
    """Create a deterministic ERA5-Land-style sample dataset."""
    rng = np.random.default_rng(seed)
    regions = [
        ("Sahara", "arid", 23.5, 13.2),
        ("East China", "monsoon", 31.2, 118.4),
        ("Amazon", "tropical", -3.4, -60.1),
        ("Central Europe", "temperate", 49.0, 10.5),
    ]
    rows: list[dict[str, object]] = []
    sample_id = 0

    for region_idx, (region, climate_zone, lat, lon) in enumerate(regions):
        for grid_idx in range(3):
            grid_lat = lat + rng.normal(0, 0.45)
            grid_lon = lon + rng.normal(0, 0.45)
            previous_target = 0.0
            for year in range(2018, 2023):
                for month in range(1, 13):
                    seasonal = np.sin(2 * np.pi * month / 12)
                    annual_drift = 0.04 * (year - 2017)
                    region_effect = (region_idx - 1.5) * 0.12
                    temperature = 0.55 * seasonal + annual_drift + rng.normal(0, 0.08)
                    precipitation = -0.35 * seasonal + rng.normal(0, 0.12)
                    radiation = 0.45 * seasonal + rng.normal(0, 0.08)
                    soil_moisture = (
                        0.55 * precipitation
                        - 0.38 * temperature
                        - 0.12 * region_idx
                        + rng.normal(0, 0.10)
                    )
                    dryness = temperature - soil_moisture
                    target = (
                        0.58 * soil_moisture
                        + 0.24 * radiation
                        - 0.18 * dryness
                        + 0.38 * previous_target
                        + region_effect
                        + rng.normal(0, 0.07)
                    )

                    rows.append(
                        {
                            "sample_id": sample_id,
                            "region": region,
                            "climate_zone": climate_zone,
                            "grid_id": f"{region.lower().replace(' ', '_')}_{grid_idx}",
                            "year": year,
                            "month": month,
                            "latitude": grid_lat,
                            "longitude": grid_lon,
                            "temperature_anomaly": temperature,
                            "precipitation_anomaly": precipitation,
                            "radiation_anomaly": radiation,
                            "soil_moisture_anomaly": soil_moisture,
                            "dryness_proxy": dryness,
                            "month_sin": np.sin(2 * np.pi * month / 12),
                            "month_cos": np.cos(2 * np.pi * month / 12),
                            f"{TARGET_COL}_lag1": previous_target,
                            TARGET_COL: target,
                        }
                    )
                    sample_id += 1
                    previous_target = target

    data = pd.DataFrame(rows)
    return data.sort_values(["region", "grid_id", "year", "month"]).reset_index(drop=True)


def build_splits(data: pd.DataFrame, seed: int = 42) -> list[SplitSpec]:
    """Build fixed sample split protocols."""
    rng = np.random.default_rng(seed)
    random_test_ids = set(rng.choice(data["sample_id"], size=int(len(data) * 0.25), replace=False))

    return [
        SplitSpec(
            name="random",
            train_mask=~data["sample_id"].isin(random_test_ids),
            test_mask=data["sample_id"].isin(random_test_ids),
        ),
        SplitSpec(
            name="temporal_holdout",
            train_mask=data["year"] <= 2020,
            test_mask=data["year"] >= 2021,
        ),
        SplitSpec(
            name="spatial_holdout",
            train_mask=~data["region"].isin(["Amazon"]),
            test_mask=data["region"].isin(["Amazon"]),
        ),
    ]


def run_sample_benchmark(output_dir: Path, seed: int = 42) -> dict[str, object]:
    """Run the fixed sample suite and write output artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_sample_dataset(seed=seed)
    splits = build_splits(data, seed=seed)

    predictions: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []

    for split in splits:
        train_df = data.loc[split.train_mask].copy()
        test_df = data.loc[split.test_mask].copy()
        event_threshold = float(train_df["soil_moisture_anomaly"].quantile(0.10))
        test_event = test_df["soil_moisture_anomaly"].to_numpy() < event_threshold

        for model_name, y_pred in _iter_model_predictions(train_df, test_df):
            y_true = test_df[TARGET_COL].to_numpy(dtype=np.float64)
            pred_event = y_pred < event_threshold
            regression = evaluate_regression(y_true, y_pred)
            detection = _compute_detection_metrics(y_true, y_pred, test_event, pred_event)

            row = {
                "model_name": model_name,
                "split_protocol": split.name,
                **regression,
                **detection,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
            }
            metrics_rows.append(row)

            pred_df = test_df[
                ["sample_id", "region", "climate_zone", "year", "month", TARGET_COL]
            ].copy()
            pred_df["model_name"] = model_name
            pred_df["split_protocol"] = split.name
            pred_df["prediction"] = y_pred
            pred_df[EVENT_COL] = test_event
            pred_df[PRED_EVENT_COL] = pred_event
            predictions.append(pred_df)

    leaderboard = _rank_metrics(pd.DataFrame(metrics_rows))
    predictions_df = pd.concat(predictions, ignore_index=True)

    data.to_csv(output_dir / "sample_dataset.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    _write_json(output_dir / "run_summary.json", _build_summary(data, leaderboard, output_dir))

    return {
        "output_dir": str(output_dir),
        "leaderboard_rows": int(len(leaderboard)),
        "prediction_rows": int(len(predictions_df)),
        "best_model": str(leaderboard.iloc[0]["model_name"]),
        "best_split": str(leaderboard.iloc[0]["split_protocol"]),
        "best_rmse": float(leaderboard.iloc[0]["rmse"]),
    }


def _iter_model_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Iterable[tuple[str, np.ndarray]]:
    models = [
        ClimatologyBaseline(target_col=TARGET_COL, predict_zero=True),
        PersistenceBaseline(target_col=TARGET_COL),
        LightGBMBaseline(
            n_estimators=20,
            learning_rate=0.08,
            num_leaves=15,
            random_state=42,
            n_jobs=1,
            force_col_wise=True,
        ),
    ]
    for model in models:
        if isinstance(model, LightGBMBaseline):
            model.fit(train_df, target_col=TARGET_COL, feature_cols=FEATURE_COLUMNS)
        else:
            model.fit(train_df, target_col=TARGET_COL)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            yield model.get_model_name(), model.predict(test_df)


def _compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_event: np.ndarray,
    pred_event: np.ndarray,
) -> dict[str, float]:
    logging.getLogger("climatenet.evaluation.detection").setLevel(logging.ERROR)
    return {
        f"{EVENT_COL}_pod": float(compute_pod(true_event, pred_event)["value"]),
        f"{EVENT_COL}_far": float(compute_far(true_event, pred_event)["value"]),
        f"{EVENT_COL}_csi": float(compute_csi(true_event, pred_event)["value"]),
        f"{EVENT_COL}_intensity_bias": float(
            compute_intensity_bias(y_true, y_pred, true_event)["value"]
        ),
    }


def _rank_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.sort_values(["rmse", "mae"], ascending=[True, True]).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _build_summary(data: pd.DataFrame, leaderboard: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    return {
        "benchmark_name": "sample_era5_land_hydroclimate",
        "data_source": "Synthetic ERA5-Land-style sample data for demonstration",
        "scientific_data_source": "ERA5-Land reanalysis",
        "target": TARGET_COL,
        "event": EVENT_COL,
        "models": sorted(leaderboard["model_name"].unique().tolist()),
        "splits": sorted(leaderboard["split_protocol"].unique().tolist()),
        "n_samples": int(len(data)),
        "regions": sorted(data["region"].unique().tolist()),
        "climate_zones": sorted(data["climate_zone"].unique().tolist()),
        "artifacts": {
            "sample_dataset": str(output_dir / "sample_dataset.csv"),
            "predictions": str(output_dir / "predictions.csv"),
            "leaderboard": str(output_dir / "leaderboard.csv"),
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ClimateNet-Bench sample benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo/sample_benchmark"),
        help="Directory for sample benchmark artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_sample_benchmark(output_dir=args.output_dir, seed=args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
