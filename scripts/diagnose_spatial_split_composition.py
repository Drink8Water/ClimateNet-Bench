#!/usr/bin/env python
"""Diagnose corrected ERA5-Land spatial split composition without training."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.diagnostics.spatial_composition import (
    assign_partitions,
    composition_tables,
    feature_shift_row,
    load_spatial_grid_assignments,
    markdown_table,
    naive_baseline_rows,
    resolve_feature_columns,
    resolve_run_dirs,
    target_distribution_rows,
)
from climatenet.utils.config import load_yaml


def _write_report(
    output: Path,
    difficulty: pd.DataFrame,
    baselines: pd.DataFrame,
    shifts: pd.DataFrame,
) -> None:
    display = difficulty.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(4)
    easiest = display.sort_values("lightgbm_spatial_rmse").iloc[0]
    other = display[display["seed"] != int(easiest["seed"])]
    baseline_view = baselines[
        (baselines["region"] == "ALL")
        & baselines["baseline"].isin(
            ["zero_anomaly", "train_target_mean", "persistence_lag_1"]
        )
    ].copy()
    baseline_view["rmse"] = baseline_view["rmse"].round(4)
    top = (
        shifts.assign(
            absolute_smd=shifts["standardized_mean_difference"].abs()
        )
        .sort_values(["seed", "absolute_smd"], ascending=[True, False])
        .groupby("seed", as_index=False)
        .head(5)
    )
    top["standardized_mean_difference"] = top[
        "standardized_mean_difference"
    ].round(4)
    top["test_outside_train_range_ratio"] = top[
        "test_outside_train_range_ratio"
    ].round(6)
    text = f"""# Corrected ERA5-Land Spatial Split Composition Diagnostic

This is a read-only diagnostic over completed corrected benchmark artifacts.
It does not fit a model or rerun a benchmark.

## Seed difficulty table

{markdown_table(display, [
    "seed", "test_sample_count", "test_grid_count",
    "east_china_test_share", "test_target_std",
    "zero_baseline_rmse", "mean_abs_feature_smd",
    "linear_spatial_rmse", "lightgbm_spatial_rmse",
    "lightgbm_skill_vs_zero_baseline"
])}

## Interpretation

Seed {int(easiest['seed'])} is the easiest observed spatial holdout for
LightGBM. Its East China test share is
{float(easiest['east_china_test_share']):.1%}, versus
{float(other['east_china_test_share'].mean()):.1%} for the other seeds.
Its zero-anomaly baseline RMSE is
{float(easiest['zero_baseline_rmse']):.3f}, versus
{float(other['zero_baseline_rmse'].mean()):.3f} on average for the other
seeds. This establishes how much of the lower model RMSE is attributable to
the held-out target mix before considering model generalisation.

LightGBM skill relative to the zero baseline is
{float(easiest['lightgbm_skill_vs_zero_baseline']):.1%} for seed
{int(easiest['seed'])}, versus
{float(other['lightgbm_skill_vs_zero_baseline'].mean()):.1%} on average for
the other seeds. Composition therefore explains a substantial part, but not
all, of the result: the particular held-out Sahara blocks are also more
predictable for LightGBM.

The mean absolute standardized feature shift for seed
{int(easiest['seed'])} is {float(easiest['mean_abs_feature_smd']):.3f}.
Compare this with {float(other['mean_abs_feature_smd'].mean()):.3f} for the
other seeds. The per-feature table should be used to distinguish easier
region/target composition from genuinely closer covariate distributions.

One spatial holdout is not sufficient for a stable spatial-generalisation
claim. Report repeated spatial folds with mean/std and publish region share,
target dispersion, baseline RMSE, held-out grid count, and feature-shift
scores alongside model metrics.

## Naive baselines

{markdown_table(baseline_view, ["seed", "baseline", "sample_count", "rmse"])}

## Largest prepared-feature shifts per seed

{markdown_table(top, [
    "seed", "feature", "standardized_mean_difference",
    "test_outside_train_range_ratio"
])}
"""
    (output / "spatial_seed_difficulty_report.md").write_text(
        text, encoding="utf-8"
    )
    (output / "spatial_baseline_difficulty_summary.md").write_text(
        "# Spatial Naive Baseline Difficulty\n\n"
        + markdown_table(
            baseline_view, ["seed", "baseline", "sample_count", "rmse"]
        )
        + "\n\nLower baseline RMSE means the held-out target distribution is "
        "intrinsically easier before any learned-model comparison.\n",
        encoding="utf-8",
    )


def run_diagnostic(config: dict[str, Any], output: Path) -> None:
    run_dirs = resolve_run_dirs(config)
    summary_root = Path(config["multi_seed_summary_path"])
    results = pd.read_csv(summary_root / "all_seed_results.csv")
    results = results[
        (results["split_protocol"] == config["target_split_id"])
        & (results["feature_set"] == config["feature_set"])
        & (results["model_name"].isin(config["models"]))
    ]
    if len(results) != len(run_dirs) * len(config["models"]):
        raise ValueError("Expected exactly one spatial metric per seed/model")

    partition_tables: list[pd.DataFrame] = []
    region_tables: list[pd.DataFrame] = []
    grid_tables: list[pd.DataFrame] = []
    month_tables: list[pd.DataFrame] = []
    target_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []

    for seed in config["seeds"]:
        seed = int(seed)
        run = run_dirs[seed]
        split_dir = run / "splits" / config["target_split_id"]
        assignments = load_spatial_grid_assignments(
            split_dir, config["regions"]
        )
        cache = (
            run
            / "preprocessing"
            / config["target_split_id"]
            / "prepared_samples.pkl"
        )
        samples = pd.read_pickle(cache)
        samples["partition"] = assign_partitions(samples, assignments)

        tables = composition_tables(samples, seed)
        partition_tables.append(tables["partition"])
        region_tables.append(tables["region"])
        grid_tables.append(tables["grid"])
        month_tables.append(tables["month"])
        target_rows.extend(target_distribution_rows(samples, seed))

        preprocessing = json.loads(
            (
                run
                / "preprocessing"
                / config["target_split_id"]
                / "preprocessing_metadata.json"
            ).read_text(encoding="utf-8")
        )
        baseline_rows.extend(
            naive_baseline_rows(
                samples,
                seed,
                preprocessing.get("standardization_parameters", {}),
            )
        )

        feature_columns = resolve_feature_columns(
            samples.columns, config["feature_patterns"]
        )
        train_mask = samples["partition"].eq("train").to_numpy()
        test_mask = samples["partition"].eq("test").to_numpy()
        for feature in feature_columns:
            values = samples[feature].to_numpy()
            shift_rows.append(
                feature_shift_row(
                    values[train_mask],
                    values[test_mask],
                    seed=seed,
                    feature=feature,
                    quantile_stride=int(
                        config.get("quantile_sample_stride", 20)
                    ),
                )
            )
        del samples, assignments, preprocessing
        gc.collect()

    partitions = pd.concat(partition_tables, ignore_index=True)
    regions = pd.concat(region_tables, ignore_index=True)
    grids = pd.concat(grid_tables, ignore_index=True)
    months = pd.concat(month_tables, ignore_index=True)
    targets = pd.DataFrame(target_rows)
    baselines = pd.DataFrame(baseline_rows)
    shifts = pd.DataFrame(shift_rows)

    partitions.to_csv(output / "split_partition_counts_by_seed.csv", index=False)
    regions.to_csv(output / "split_region_counts_by_seed.csv", index=False)
    grids.to_csv(output / "split_grid_counts_by_seed.csv", index=False)
    months.to_csv(output / "split_month_counts_by_seed.csv", index=False)
    targets.to_csv(
        output / "target_distribution_by_seed_partition_region.csv",
        index=False,
    )
    baselines.to_csv(output / "spatial_naive_baselines_by_seed.csv", index=False)
    shifts.to_csv(output / "spatial_feature_shift_by_seed.csv", index=False)
    (
        shifts.assign(
            absolute_standardized_mean_difference=shifts[
                "standardized_mean_difference"
            ].abs()
        )
        .sort_values(
            ["seed", "absolute_standardized_mean_difference"],
            ascending=[True, False],
        )
        .groupby("seed", as_index=False)
        .head(10)
        .to_csv(output / "spatial_feature_shift_top_features.csv", index=False)
    )

    overall_targets = targets[targets["region"] == "ALL"].set_index(
        ["seed", "partition"]
    )
    shift_summary = []
    for seed in config["seeds"]:
        train = overall_targets.loc[(int(seed), "train")]
        test = overall_targets.loc[(int(seed), "test")]
        shift_summary.append(
            {
                "seed": int(seed),
                "train_target_mean": train["mean"],
                "test_target_mean": test["mean"],
                "train_target_std": train["std"],
                "test_target_std": test["std"],
                "train_vs_test_target_std_ratio": (
                    test["std"] / train["std"]
                ),
                "train_vs_test_target_mean_shift": (
                    test["mean"] - train["mean"]
                ),
                "train_zero_baseline_rmse": train[
                    "zero_anomaly_baseline_rmse"
                ],
                "test_zero_baseline_rmse": test[
                    "zero_anomaly_baseline_rmse"
                ],
            }
        )
    target_shift = pd.DataFrame(shift_summary)
    target_shift.to_csv(
        output / "target_distribution_shift_summary.csv", index=False
    )

    metric_pivot = results.pivot(
        index="seed", columns="model_name", values="rmse"
    ).rename(
        columns={
            "linear_regression": "linear_spatial_rmse",
            "lightgbm": "lightgbm_spatial_rmse",
        }
    )
    test_regions = regions[regions["partition"] == "test"]
    east_share = (
        test_regions[test_regions["region"].astype(str) == "East China"]
        .set_index("seed")["partition_share"]
        .rename("east_china_test_share")
    )
    test_counts = (
        partitions[partitions["partition"] == "test"]
        .set_index("seed")["sample_count"]
        .rename("test_sample_count")
    )
    test_grids = (
        grids[grids["partition"] == "test"]
        .set_index("seed")["unique_grid_cell_count"]
        .rename("test_grid_count")
    )
    feature_summary = shifts.groupby("seed").agg(
        mean_abs_feature_smd=(
            "standardized_mean_difference",
            lambda values: float(np.mean(np.abs(values))),
        ),
        max_abs_feature_smd=(
            "standardized_mean_difference",
            lambda values: float(np.max(np.abs(values))),
        ),
        mean_train_range_exceedance=(
            "test_outside_train_range_ratio",
            "mean",
        ),
    )
    baseline_overall = (
        baselines[
            (baselines["region"] == "ALL")
            & (baselines["baseline"] == "zero_anomaly")
        ]
        .set_index("seed")["rmse"]
        .rename("zero_baseline_rmse")
    )
    target_test = (
        target_shift.set_index("seed")[
            [
                "test_target_mean",
                "test_target_std",
                "train_vs_test_target_std_ratio",
                "train_vs_test_target_mean_shift",
            ]
        ]
    )
    difficulty = pd.concat(
        [
            test_counts,
            test_grids,
            east_share,
            target_test,
            baseline_overall,
            feature_summary,
            metric_pivot,
        ],
        axis=1,
    ).reset_index()
    difficulty["linear_skill_vs_zero_baseline"] = (
        1.0
        - difficulty["linear_spatial_rmse"]
        / difficulty["zero_baseline_rmse"]
    )
    difficulty["lightgbm_skill_vs_zero_baseline"] = (
        1.0
        - difficulty["lightgbm_spatial_rmse"]
        / difficulty["zero_baseline_rmse"]
    )
    difficulty.to_csv(output / "spatial_seed_difficulty_table.csv", index=False)
    _write_report(output, difficulty, baselines, shifts)

    manifest = {
        "status": "completed",
        "diagnostic_name": config["diagnostic_name"],
        "run_dirs": {str(seed): str(path) for seed, path in run_dirs.items()},
        "physical_features_path": config["physical_features_path"],
        "physical_features_read": False,
        "data_access": (
            "Prepared split caches, split ID manifests, preprocessing "
            "metadata, and completed metrics only"
        ),
        "models_retrained": False,
        "benchmark_rerun": False,
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    (output / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/diagnostics/"
            "era5_land_corrected_spatial_split_composition.yaml"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(config["output_dir"])
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        run_diagnostic(config, output)
    except Exception:
        (output / "diagnostic_status.json").write_text(
            json.dumps({"status": "failed"}, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
