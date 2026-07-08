"""Tests for the sample benchmark demo script."""

from __future__ import annotations

import json

import pandas as pd

from scripts.demo_smoke import run_sample_benchmark


def test_demo_smoke_writes_sample_benchmark_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "sample_benchmark"

    summary = run_sample_benchmark(output_dir=output_dir, seed=7)

    leaderboard_path = output_dir / "leaderboard.csv"
    predictions_path = output_dir / "predictions.csv"
    dataset_path = output_dir / "sample_dataset.csv"
    summary_path = output_dir / "run_summary.json"

    assert leaderboard_path.exists()
    assert predictions_path.exists()
    assert dataset_path.exists()
    assert summary_path.exists()

    leaderboard = pd.read_csv(leaderboard_path)
    predictions = pd.read_csv(predictions_path)
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert set(leaderboard["model_name"]) == {"climatology", "persistence", "lightgbm"}
    assert set(leaderboard["split_protocol"]) == {
        "random",
        "temporal_holdout",
        "spatial_holdout",
    }
    assert leaderboard["rank"].tolist() == list(range(1, len(leaderboard) + 1))
    assert leaderboard["rmse"].notna().all()
    assert "soil_moisture_drought_csi" in leaderboard.columns
    assert {"evaporation_anomaly", "prediction", "soil_moisture_drought"}.issubset(
        predictions.columns
    )
    assert run_summary["scientific_data_source"] == "ERA5-Land reanalysis"
    assert summary["leaderboard_rows"] == 9
