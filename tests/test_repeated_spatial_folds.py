from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climatenet.benchmark.manifest_split_adapter import (
    load_manifest_backed_repeated_splits,
)
from climatenet.benchmark.repeated_spatial import (
    RepeatedSpatialDesign,
    add_spatial_block_id,
    assign_row_partitions,
    evaluate_fold_acceptance,
    fit_train_only_target_anomaly,
    fold_block_assignments,
    generate_block_test_folds,
    unique_grid_table,
    validate_fold_isolation,
)


def _grid_frame() -> pd.DataFrame:
    rows = []
    for region, lon_offset in [("Sahara", 0), ("East China", 100)]:
        for block in range(5):
            for cell in range(block + 1):
                rows.append(
                    {
                        "region": region,
                        "latitude": block * 5 + cell * 0.1,
                        "longitude": lon_offset + cell * 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_repeated_folds_are_deterministic_and_region_stratified() -> None:
    grids = unique_grid_table(_grid_frame(), 5.0)
    design = RepeatedSpatialDesign(fold_count=5, random_seed=42)
    first = generate_block_test_folds(grids, design)
    second = generate_block_test_folds(grids, design)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.groupby("region")["test_fold"].nunique()) == {5}

    roles = []
    for fold in range(5):
        assignments = fold_block_assignments(first, fold, 5)
        for partition in ["train", "validation", "test"]:
            assert set(
                assignments.loc[
                    assignments["partition"] == partition, "region"
                ]
            ) == {"Sahara", "East China"}
        roles.append(assignments[["region", "spatial_block_id", "partition"]])
    combined = pd.concat(roles)
    counts = combined.groupby(["region", "spatial_block_id", "partition"]).size()
    assert set(counts.xs("test", level="partition")) == {1}
    assert set(counts.xs("validation", level="partition")) == {1}
    assert set(counts.xs("train", level="partition")) == {3}


def test_all_months_for_grid_follow_one_partition() -> None:
    base = _grid_frame()
    grids = unique_grid_table(base, 5.0)
    test_folds = generate_block_test_folds(
        grids, RepeatedSpatialDesign(fold_count=5)
    )
    assignments = fold_block_assignments(test_folds, 0, 5)
    rows = pd.concat(
        [base.assign(month=month) for month in range(1, 13)],
        ignore_index=True,
    )
    rows = add_spatial_block_id(rows, 5.0)
    partition = assign_row_partitions(rows, assignments)
    rows["partition"] = partition
    assert (
        rows.groupby(["region", "latitude", "longitude"])["partition"]
        .nunique()
        .max()
        == 1
    )
    isolation = validate_fold_isolation(grids, assignments)
    assert isolation["grid_leakage_count"] == 0
    assert isolation["duplicate_assignment_count"] == 0


def test_target_climatology_is_train_only() -> None:
    frame = pd.DataFrame(
        {
            "region": ["Sahara"] * 4,
            "month": [1, 1, 1, 1],
            "evaporation": [1.0, 3.0, 10.0, 20.0],
        }
    )
    partition = pd.Series(["train", "train", "validation", "test"])
    anomaly, climatology = fit_train_only_target_anomaly(frame, partition)
    changed = frame.copy()
    changed.loc[changed.index[-1], "evaporation"] = 20_000.0
    changed_anomaly, changed_climatology = fit_train_only_target_anomaly(
        changed, partition
    )
    pd.testing.assert_frame_equal(climatology, changed_climatology)
    np.testing.assert_allclose(anomaly[:2], changed_anomaly[:2])
    assert climatology["train_climatology"].iloc[0] == pytest.approx(2.0)


def test_acceptance_flags_imbalanced_fold() -> None:
    rows = pd.DataFrame(
        {
            "fold": [0, 1],
            "test_east_china_share": [0.25, 0.05],
            "validation_east_china_share": [0.25, 0.25],
            "test_min_region_grid_count": [1, 1],
            "validation_min_region_grid_count": [1, 1],
            "grid_leakage_count": [0, 0],
            "duplicate_assignment_count": [0, 0],
            "train_month_count": [12, 12],
            "validation_month_count": [12, 12],
            "test_month_count": [12, 12],
            "test_target_std": [10.0, 10.0],
            "zero_baseline_rmse": [10.0, 10.0],
        }
    )
    acceptance = {
        "min_test_east_china_share": 0.15,
        "max_test_east_china_share": 0.60,
        "min_validation_east_china_share": 0.15,
        "max_validation_east_china_share": 0.60,
        "min_target_std_vs_fold_median": 0.85,
        "max_zero_baseline_rmse_cv": 0.15,
    }
    evaluated, summary = evaluate_fold_acceptance(
        rows, acceptance=acceptance
    )
    assert evaluated["fold_passed"].tolist() == [True, False]
    assert summary["audit_passed"] is False


def test_manifest_adapter_builds_audited_fold_split(tmp_path) -> None:
    samples = pd.DataFrame(
        {
            "sample_id": [f"sample-{grid}-{month}" for grid in range(3)
                          for month in range(2)],
            "grid_id": [f"grid-{grid}" for grid in range(3)
                        for _ in range(2)],
            "region": ["Sahara"] * 6,
            "latitude": [0.0, 0.0, 10.0, 10.0, 20.0, 20.0],
            "longitude": [0.0] * 6,
        }
    )
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    pd.DataFrame(
        {
            "region": ["Sahara"] * 3,
            "spatial_block_id": [
                "block_lat0_lon0",
                "block_lat10_lon0",
                "block_lat20_lon0",
            ],
            "partition": ["train", "validation", "test"],
        }
    ).to_csv(manifest_dir / "fold_0_block_assignments.csv", index=False)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        '{"status":"ready","audit_passed":true,'
        '"folds":[{"fold":0,"fold_passed":true}]}',
        encoding="utf-8",
    )
    results = load_manifest_backed_repeated_splits(
        samples,
        {
            "audit_path": str(audit_path),
            "manifest_dir": str(manifest_dir),
            "required_audit_status": "ready",
            "folds": [0],
            "block_size_deg": 5.0,
            "generation_seed": 42,
        },
        tmp_path / "splits",
    )
    result = results[0]
    assert result.split_id == "repeated_region_stratified_spatial_fold_0"
    assert result.metadata["assignment_uses_target"] is False
    assert result.metadata["fold_audit_status"] == "ready"
    assert len(result.train_ids) == len(result.val_ids) == len(
        result.test_ids
    ) == 2
