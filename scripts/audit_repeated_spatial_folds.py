#!/usr/bin/env python
"""Generate and audit repeated region-stratified spatial folds.

This script is read-only with respect to source data and does not train a
model or invoke the benchmark runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.benchmark.repeated_spatial import (
    PARTITIONS,
    RepeatedSpatialDesign,
    add_spatial_block_id,
    array_summary,
    assign_row_partitions,
    evaluate_fold_acceptance,
    fit_train_only_target_anomaly,
    fold_block_assignments,
    generate_block_test_folds,
    unique_grid_table,
    validate_fold_isolation,
)
from climatenet.diagnostics.spatial_composition import (
    feature_shift_row,
    markdown_table,
)
from climatenet.utils.config import load_yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_required_columns(config: dict[str, Any]) -> pd.DataFrame:
    columns = config["columns"]
    usecols = [
        columns["region"],
        columns["year"],
        columns["month"],
        columns["latitude"],
        columns["longitude"],
        columns["target_source"],
        *config["feature_shift_columns"],
    ]
    usecols = list(dict.fromkeys(usecols))
    pieces: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(
        config["input_path"],
        usecols=usecols,
        chunksize=int(config["chunksize"]),
    ):
        total += len(chunk)
        if total > int(config["safety_max_rows"]):
            raise ValueError(
                f"Input exceeds safety_max_rows={config['safety_max_rows']}"
            )
        numeric = [column for column in usecols if column != columns["region"]]
        values = chunk[numeric].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Input contains NaN or Inf in required columns")
        for column in numeric:
            if column == columns["target_source"]:
                chunk[column] = chunk[column].astype(np.float64)
            elif column in (columns["year"], columns["month"]):
                continue
            else:
                chunk[column] = chunk[column].astype(np.float32)
        pieces.append(chunk)
    frame = pd.concat(pieces, ignore_index=True)
    frame[columns["region"]] = frame[columns["region"]].astype("category")
    frame[columns["year"]] = frame[columns["year"]].astype(np.int16)
    frame[columns["month"]] = frame[columns["month"]].astype(np.int8)
    return frame


def _target_distribution(
    frame: pd.DataFrame,
    partition: pd.Series,
    anomaly: np.ndarray,
    fold: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    regions = frame["region"].astype("string").to_numpy()
    for role in PARTITIONS:
        role_mask = partition.eq(role).to_numpy()
        rows.append(
            {
                "fold": fold,
                "partition": role,
                "region": "ALL",
                **array_summary(anomaly[role_mask]),
            }
        )
        for region in sorted(frame["region"].astype(str).unique()):
            mask = role_mask & (regions == region)
            rows.append(
                {
                    "fold": fold,
                    "partition": role,
                    "region": region,
                    **array_summary(anomaly[mask]),
                }
            )
    return rows


def _naive_baselines(
    frame: pd.DataFrame,
    partition: pd.Series,
    anomaly: np.ndarray,
    fold: int,
) -> list[dict[str, Any]]:
    train_mask = partition.eq("train").to_numpy()
    test_mask = partition.eq("test").to_numpy()
    regions = frame["region"].astype("string").to_numpy()
    months = frame["month"].to_numpy()
    train_mean = float(anomaly[train_mask].mean())
    region_month_mean: dict[tuple[str, int], float] = {}
    for region in sorted(frame["region"].astype(str).unique()):
        for month in range(1, 13):
            mask = train_mask & (regions == region) & (months == month)
            region_month_mean[(region, month)] = float(anomaly[mask].mean())
    region_month_prediction = np.array(
        [
            region_month_mean[(str(region), int(month))]
            for region, month in zip(regions[test_mask], months[test_mask])
        ],
        dtype=np.float64,
    )
    truth_all = anomaly[test_mask]
    predictions = {
        "zero_anomaly": np.zeros(len(truth_all)),
        "train_target_mean": np.full(len(truth_all), train_mean),
        "train_region_month_target_mean": region_month_prediction,
    }
    test_regions = regions[test_mask]
    rows: list[dict[str, Any]] = []
    for region in ["ALL", *sorted(np.unique(test_regions))]:
        mask = (
            np.ones(len(truth_all), dtype=bool)
            if region == "ALL"
            else test_regions == region
        )
        for baseline, prediction in predictions.items():
            rmse = float(
                np.sqrt(
                    np.mean(np.square(truth_all[mask] - prediction[mask]))
                )
            )
            rows.append(
                {
                    "fold": fold,
                    "region": region,
                    "baseline": baseline,
                    "sample_count": int(mask.sum()),
                    "rmse": rmse,
                    "fit_scope": (
                        "none"
                        if baseline == "zero_anomaly"
                        else "train_only"
                    ),
                }
            )
    return rows


def _write_report(
    output: Path,
    balance: pd.DataFrame,
    global_summary: dict[str, Any],
    block_coverage: pd.DataFrame,
) -> None:
    display = balance[
        [
            "fold",
            "test_east_china_share",
            "validation_east_china_share",
            "test_target_std",
            "zero_baseline_rmse",
            "grid_leakage_count",
            "fold_passed",
        ]
    ].copy()
    for column in [
        "test_east_china_share",
        "validation_east_china_share",
        "test_target_std",
        "zero_baseline_rmse",
    ]:
        display[column] = display[column].round(4)
    coverage_ok = bool(
        (block_coverage["test_count"] == 1).all()
        and (block_coverage["validation_count"] == 1).all()
        and (block_coverage["train_count"] == 3).all()
    )
    status = "ready" if global_summary["audit_passed"] else "not_ready"
    report = f"""# Repeated Region-Stratified Spatial Fold Audit

Status: **{status}**

Five folds were generated independently within each region using 5-degree
spatial blocks. A latitude-band snake traversal interleaves blocks across
folds without inspecting targets or features. Fold `k` uses bin `k` for test
and bin `k+1` for validation. All months follow their grid cell.

{markdown_table(display, list(display.columns))}

## Leakage and coverage

- Grid leakage: `{int(balance['grid_leakage_count'].sum())}`
- Duplicate block assignments: `{int(balance['duplicate_assignment_count'].sum())}`
- Every partition covers all 12 calendar months: `{bool(
    (balance[['train_month_count', 'validation_month_count', 'test_month_count']] == 12)
    .all()
    .all()
)}`
- Every block is test once, validation once, and train three times:
  `{coverage_ok}`

## Balance criteria

- Test East China share: 15%-60%.
- Validation East China share: 15%-60%.
- Both regions must have non-zero validation and test grids.
- No grid leakage, duplicate assignment, or missing month.
- Test target std must be at least 85% of the five-fold median.
- Zero-baseline RMSE CV must not exceed 0.15.

Observed zero-baseline RMSE CV:
`{global_summary['zero_baseline_rmse_cv']:.4f}`.

Target anomalies were fitted separately for every fold from train-only
`region x month` evaporation climatology. Validation and test values never
contributed to the target climatology.

This is a composition audit only. Repeated spatial models have **not** been
trained or benchmarked.
"""
    (output / "repeated_spatial_folds_audit.md").write_text(
        report, encoding="utf-8"
    )


def run_audit(config: dict[str, Any], output: Path) -> dict[str, Any]:
    input_path = Path(config["input_path"])
    actual_hash = _sha256(input_path)
    if actual_hash != config["expected_input_sha256"]:
        raise ValueError(
            f"Input SHA256 mismatch: {actual_hash} != "
            f"{config['expected_input_sha256']}"
        )
    split_config = load_yaml(config["split_config_path"])
    design = RepeatedSpatialDesign(
        fold_count=int(split_config["fold_count"]),
        block_size_deg=float(split_config["block_size_deg"]),
        random_seed=int(split_config["random_seed"]),
        assignment_method=(
            "geographic_interleaving"
            if split_config["assignment_method"].startswith(
                "geographic_interleaving"
            )
            else split_config["assignment_method"]
        ),
    )
    frame = _load_required_columns(config)
    frame = frame.rename(
        columns={
            config["columns"]["region"]: "region",
            config["columns"]["year"]: "year",
            config["columns"]["month"]: "month",
            config["columns"]["latitude"]: "latitude",
            config["columns"]["longitude"]: "longitude",
            config["columns"]["target_source"]: "evaporation",
        }
    )
    frame = add_spatial_block_id(frame, design.block_size_deg)
    grids = unique_grid_table(frame, design.block_size_deg)
    test_folds = generate_block_test_folds(grids, design)
    test_folds.to_csv(output / "block_test_fold_assignments.csv", index=False)
    manifests = output / "fold_manifests"
    manifests.mkdir(exist_ok=True)

    partition_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    coverage_rows: list[pd.DataFrame] = []

    for fold in range(design.fold_count):
        assignments = fold_block_assignments(
            test_folds, fold, design.fold_count
        )
        assignments.to_csv(
            manifests / f"fold_{fold}_block_assignments.csv", index=False
        )
        isolation = validate_fold_isolation(grids, assignments)
        partition = assign_row_partitions(frame, assignments)

        counts = partition.value_counts()
        for role in PARTITIONS:
            partition_rows.append(
                {
                    "fold": fold,
                    "partition": role,
                    "sample_count": int(counts.get(role, 0)),
                    "sample_share": float(counts.get(role, 0) / len(frame)),
                }
            )
        regions = frame["region"].astype("string")
        months = frame["month"]
        for role in PARTITIONS:
            role_mask = partition.eq(role)
            role_total = int(role_mask.sum())
            for region in sorted(regions.unique()):
                sample_count = int((role_mask & regions.eq(region)).sum())
                region_rows.append(
                    {
                        "fold": fold,
                        "partition": role,
                        "region": region,
                        "sample_count": sample_count,
                        "partition_share": sample_count / role_total,
                    }
                )
            assigned_blocks = assignments[
                assignments["partition"] == role
            ]
            for region in sorted(regions.unique()):
                regional = assigned_blocks[
                    assigned_blocks["region"] == region
                ]
                grid_rows.append(
                    {
                        "fold": fold,
                        "partition": role,
                        "region": region,
                        "unique_grid_cell_count": int(
                            regional["grid_count"].sum()
                        ),
                        "spatial_block_count": int(len(regional)),
                    }
                )
        anomaly, climatology = fit_train_only_target_anomaly(
            frame, partition, value_column="evaporation"
        )
        climatology.to_csv(
            manifests / f"fold_{fold}_train_climatology.csv", index=False
        )
        fold_targets = _target_distribution(
            frame, partition, anomaly, fold
        )
        target_rows.extend(fold_targets)
        baseline_rows.extend(
            _naive_baselines(frame, partition, anomaly, fold)
        )

        train_mask = partition.eq("train").to_numpy()
        test_mask = partition.eq("test").to_numpy()
        for feature in config["feature_shift_columns"]:
            values = frame[feature].to_numpy()
            feature_rows.append(
                feature_shift_row(
                    values[train_mask],
                    values[test_mask],
                    seed=fold,
                    feature=feature,
                    quantile_stride=int(
                        config.get("quantile_sample_stride", 20)
                    ),
                )
                | {"fold": fold}
            )

        fold_target = pd.DataFrame(fold_targets)
        overall_test = fold_target[
            (fold_target["partition"] == "test")
            & (fold_target["region"] == "ALL")
        ].iloc[0]
        train_overall = fold_target[
            (fold_target["partition"] == "train")
            & (fold_target["region"] == "ALL")
        ].iloc[0]
        region_frame = pd.DataFrame(region_rows)
        test_region = region_frame[
            (region_frame["fold"] == fold)
            & (region_frame["partition"] == "test")
        ]
        val_region = region_frame[
            (region_frame["fold"] == fold)
            & (region_frame["partition"] == "validation")
        ]
        grid_frame = pd.DataFrame(grid_rows)
        test_grid = grid_frame[
            (grid_frame["fold"] == fold)
            & (grid_frame["partition"] == "test")
        ]
        val_grid = grid_frame[
            (grid_frame["fold"] == fold)
            & (grid_frame["partition"] == "validation")
        ]
        east_test = float(
            test_region.loc[
                test_region["region"] == "East China", "partition_share"
            ].iloc[0]
        )
        east_val = float(
            val_region.loc[
                val_region["region"] == "East China", "partition_share"
            ].iloc[0]
        )
        balance_rows.append(
            {
                "fold": fold,
                "test_east_china_share": east_test,
                "validation_east_china_share": east_val,
                "test_min_region_grid_count": int(
                    test_grid["unique_grid_cell_count"].min()
                ),
                "validation_min_region_grid_count": int(
                    val_grid["unique_grid_cell_count"].min()
                ),
                **isolation,
                "train_month_count": int(
                    months[partition.eq("train")].nunique()
                ),
                "validation_month_count": int(
                    months[partition.eq("validation")].nunique()
                ),
                "test_month_count": int(
                    months[partition.eq("test")].nunique()
                ),
                "train_target_std": float(train_overall["std"]),
                "test_target_std": float(overall_test["std"]),
                "train_test_target_mean_shift": float(
                    overall_test["mean"] - train_overall["mean"]
                ),
                "zero_baseline_rmse": float(
                    overall_test["zero_baseline_rmse"]
                ),
            }
        )
        coverage = assignments[
            ["region", "spatial_block_id", "partition"]
        ].copy()
        coverage["fold"] = fold
        coverage_rows.append(coverage)
        (manifests / f"fold_{fold}_manifest.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "protocol": split_config["protocol"],
                    "block_size_deg": design.block_size_deg,
                    "partition_sample_counts": {
                        role: int(counts.get(role, 0))
                        for role in PARTITIONS
                    },
                    "train_only_target_climatology": True,
                    **isolation,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    partitions = pd.DataFrame(partition_rows)
    regions = pd.DataFrame(region_rows)
    grid_counts = pd.DataFrame(grid_rows)
    targets = pd.DataFrame(target_rows)
    baselines = pd.DataFrame(baseline_rows)
    features = pd.DataFrame(feature_rows).drop(columns=["seed"])
    balance = pd.DataFrame(balance_rows)
    balance, global_summary = evaluate_fold_acceptance(
        balance, acceptance=config["acceptance"]
    )
    coverage = pd.concat(coverage_rows, ignore_index=True)
    block_coverage = (
        coverage.groupby(["region", "spatial_block_id", "partition"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(
            columns={
                "train": "train_count",
                "validation": "validation_count",
                "test": "test_count",
            }
        )
    )

    partitions.to_csv(output / "fold_partition_counts.csv", index=False)
    regions.to_csv(output / "fold_region_share.csv", index=False)
    grid_counts.to_csv(output / "fold_grid_counts.csv", index=False)
    targets.to_csv(output / "fold_target_distribution.csv", index=False)
    baselines.to_csv(output / "fold_naive_baselines.csv", index=False)
    features.to_csv(output / "fold_feature_shift_summary.csv", index=False)
    block_coverage.to_csv(output / "fold_block_coverage.csv", index=False)
    balance.to_csv(output / "fold_acceptance.csv", index=False)
    summary = {
        "status": "ready" if global_summary["audit_passed"] else "not_ready",
        "audit_passed": global_summary["audit_passed"],
        "input_path": str(input_path),
        "input_sha256": actual_hash,
        "input_rows": int(len(frame)),
        "input_columns_read": [
            "region",
            "year",
            "month",
            "latitude",
            "longitude",
            "evaporation",
            *config["feature_shift_columns"],
        ],
        "fold_count": design.fold_count,
        "block_size_deg": design.block_size_deg,
        "target_workflow": (
            "per-fold train-only region-month evaporation climatology"
        ),
        "benchmark_executed": False,
        "models_trained": False,
        "physical_csv_read_mode": "selected columns with chunking",
        "global_acceptance": global_summary,
        "folds": balance.to_dict(orient="records"),
    }
    (output / "fold_balance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(output, balance, global_summary, block_coverage)
    (output / "integration_plan.md").write_text(
        """# Repeated Spatial Runner Integration Plan

The current benchmark runner accepts one `SplitResult` per configured
protocol and does not consume a directory of precomputed fold manifests.
Keep this audit splitter independent for now.

Recommended bounded adapter:

1. Add a manifest-backed split loader that reads one audited fold assignment.
2. Invoke the existing train-only preprocessing path independently per fold.
3. Create a fresh model for each model x fold task.
4. Preserve fold ID in experiment IDs, predictions, metrics, and metadata.
5. Aggregate the ten tasks by model using fold mean/std.

No runner change or benchmark execution was performed by this audit.
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/diagnostics/"
            "era5_land_repeated_spatial_folds_audit.yaml"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(config["output_dir"])
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_audit(config, output)
    except Exception as exc:
        (output / "audit_status.json").write_text(
            json.dumps(
                {"status": "failed", "error": str(exc)},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
