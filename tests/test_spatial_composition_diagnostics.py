from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.diagnostics.spatial_composition import (
    assign_partitions,
    feature_shift_row,
    load_spatial_grid_assignments,
    naive_baseline_rows,
    resolve_run_dirs,
    target_summary,
)


def _write_split(path: Path, partition: str, ids: list[str]) -> None:
    name = "val" if partition == "validation" else partition
    pd.DataFrame({"sample_id": ids}).to_csv(
        path / f"{name}_ids.csv", index=False
    )


def test_assignment_uses_region_and_grid_not_target_month(tmp_path: Path) -> None:
    _write_split(
        tmp_path, "train", ["Sahara_10.0000_20.0000_2019_07"]
    )
    _write_split(
        tmp_path, "validation", ["Sahara_11.0000_20.0000_2019_07"]
    )
    _write_split(
        tmp_path, "test", ["East China_20.0000_110.0000_2019_07"]
    )
    assignments = load_spatial_grid_assignments(
        tmp_path, ["Sahara", "East China"]
    )
    samples = pd.DataFrame(
        {
            "sample_id": [
                "Sahara_10.0000_20.0000_2023_12",
                "Sahara_11.0000_20.0000_2022_06",
                "East China_20.0000_110.0000_2021_01",
            ],
            "region": ["Sahara", "Sahara", "East China"],
            "grid_id": [
                "10.0000_20.0000",
                "11.0000_20.0000",
                "20.0000_110.0000",
            ],
        }
    )
    assert assign_partitions(samples, assignments).tolist() == [
        "train",
        "validation",
        "test",
    ]


def test_target_summary_detects_intrinsic_baseline_difficulty() -> None:
    easy = target_summary(pd.Series([-1.0, 1.0]))
    hard = target_summary(pd.Series([-4.0, 4.0]))
    assert easy["std"] == pytest.approx(1.0)
    assert easy["zero_anomaly_baseline_rmse"] == pytest.approx(1.0)
    assert hard["zero_anomaly_baseline_rmse"] > easy[
        "zero_anomaly_baseline_rmse"
    ]


def test_naive_baselines_use_train_only_and_inverse_persistence() -> None:
    frame = pd.DataFrame(
        {
            "partition": ["train", "train", "test", "test"],
            "region": ["Sahara"] * 4,
            "target_month": [1, 2, 1, 2],
            "y_true": [2.0, 4.0, 3.0, 5.0],
            "evaporation_anomaly_lag_1": [0.0, 1.0, 0.5, 1.5],
        }
    )
    rows = naive_baseline_rows(
        frame,
        42,
        {
            "evaporation_anomaly_lag_1": {
                "mean": 2.0,
                "scale": 2.0,
            }
        },
    )
    overall = {
        row["baseline"]: row["rmse"]
        for row in rows
        if row["region"] == "ALL"
    }
    assert overall["train_target_mean"] == pytest.approx(np.sqrt(2.0))
    assert overall["train_region_month_target_mean"] == pytest.approx(1.0)
    assert overall["persistence_lag_1"] == pytest.approx(0.0)


def test_naive_baseline_uses_raw_persistence_when_not_standardized() -> None:
    frame = pd.DataFrame(
        {
            "partition": ["train", "train", "test", "test"],
            "region": ["Sahara"] * 4,
            "target_month": [1, 2, 1, 2],
            "y_true": [0.0, 0.0, 3.0, 5.0],
            "evaporation_anomaly_lag_1": [0.0, 0.0, 2.0, 4.0],
        }
    )
    rows = naive_baseline_rows(frame, 42, {})
    persistence = next(
        row for row in rows
        if row["region"] == "ALL"
        and row["baseline"] == "persistence_lag_1"
    )
    assert persistence["rmse"] == pytest.approx(1.0)


def test_feature_shift_reports_smd_and_range_exceedance() -> None:
    row = feature_shift_row(
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 3.0]),
        seed=2026,
        feature="temperature_anomaly_lag_1",
        quantile_stride=1,
    )
    assert row["standardized_mean_difference"] > 1.0
    assert row["test_outside_train_range_ratio"] == pytest.approx(2 / 3)


def test_run_paths_must_match_authoritative_summary(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_metadata.json").write_text(
        json.dumps({"status": "completed", "seed": 42}), encoding="utf-8"
    )
    summary = tmp_path / "summary"
    summary.mkdir()
    (summary / "multiseed_summary.json").write_text(
        json.dumps({"run_dirs": {"42": str(run)}}), encoding="utf-8"
    )
    config = {
        "multi_seed_summary_path": str(summary),
        "seeds": [42],
        "run_dirs": {"42": str(tmp_path / "wrong")},
    }
    with pytest.raises(ValueError, match="do not exactly match"):
        resolve_run_dirs(config)
