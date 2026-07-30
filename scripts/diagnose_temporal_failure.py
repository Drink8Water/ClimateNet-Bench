#!/usr/bin/env python
"""Generate bounded temporal-failure diagnostics from completed run artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.diagnostics.temporal_failure import (
    RunningStats,
    SEASON_BY_MONTH,
    aggregate_prediction_metrics,
    compute_error_metrics,
    shift_row,
    validate_repeated_spatial_plan,
    write_json,
)
from climatenet.utils.config import load_yaml, save_yaml


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free, Markdown-compatible CSV code block."""
    return f"```csv\n{frame.to_csv(index=False).strip()}\n```"


def _git_snapshot(output: Path) -> None:
    root = output / "code_snapshot"
    root.mkdir(exist_ok=True)
    commands = {
        "git_status_short.txt": ["git", "status", "--short"],
        "git_diff_stat.txt": ["git", "diff", "--stat", "HEAD"],
        "git_diff.patch": ["git", "diff", "--binary", "HEAD"],
        "git_commit.txt": ["git", "rev-parse", "HEAD"],
    }
    for filename, command in commands.items():
        result = subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        (root / filename).write_text(result.stdout, encoding="utf-8")


def _partition(year: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                year.isin([2019, 2020, 2021]),
                year.eq(2022),
                year.eq(2023),
            ],
            ["train", "validation", "test"],
            default="outside",
        ),
        index=year.index,
    )


def _stats_summary(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "missing_count": int(values.isna().sum()),
        "zero_anomaly_baseline_rmse": float(
            np.sqrt(np.mean(np.square(array)))
        ),
    }


def _prediction_diagnostics(
    config: dict[str, Any], output: Path
) -> dict[str, Any]:
    per_year: list[pd.DataFrame] = []
    per_month: list[pd.DataFrame] = []
    per_season: list[pd.DataFrame] = []
    per_region_year: list[pd.DataFrame] = []
    residual_summary: list[dict[str, Any]] = []
    input_files: list[dict[str, Any]] = []
    for seed_value, run_value in config["run_dirs"].items():
        seed = int(seed_value)
        run = Path(run_value)
        for model in config["models"]:
            matches = list(
                (run / "predictions").glob(
                    f"*_{model}_temporal_full_seed{seed}.csv"
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one temporal prediction for seed={seed}, "
                    f"model={model}; found {matches}"
                )
            path = matches[0]
            columns = [
                "y_true",
                "y_pred",
                "partition",
                "region",
                "target_year",
                "target_month",
            ]
            frame = pd.read_csv(path, usecols=columns)
            frame["seed"] = seed
            frame["model_name"] = model
            frame["season"] = frame["target_month"].map(SEASON_BY_MONTH)
            base = ["seed", "model_name"]
            per_year.append(
                aggregate_prediction_metrics(
                    frame, [*base, "target_year"]
                )
            )
            per_month.append(
                aggregate_prediction_metrics(
                    frame, [*base, "target_month"]
                )
            )
            per_season.append(
                aggregate_prediction_metrics(frame, [*base, "season"])
            )
            per_region_year.append(
                aggregate_prediction_metrics(
                    frame, [*base, "region", "target_year"]
                )
            )
            residual_summary.append(
                {
                    "seed": seed,
                    "model_name": model,
                    **compute_error_metrics(frame),
                }
            )
            input_files.append(
                {
                    "seed": seed,
                    "model_name": model,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "columns_read": columns,
                }
            )
            del frame
    pd.concat(per_year, ignore_index=True).to_csv(
        output / "per_year_metrics.csv", index=False
    )
    pd.concat(per_month, ignore_index=True).to_csv(
        output / "per_month_metrics.csv", index=False
    )
    pd.concat(per_season, ignore_index=True).to_csv(
        output / "per_season_metrics.csv", index=False
    )
    pd.concat(per_region_year, ignore_index=True).to_csv(
        output / "per_region_year_metrics.csv", index=False
    )
    write_json(
        output / "residual_distribution_summary.json",
        {"rows": residual_summary},
    )
    return {
        "prediction_files": input_files,
        "total_bytes_read": sum(item["size_bytes"] for item in input_files),
    }


def _update_raw_stats(
    store: dict[tuple[Any, ...], RunningStats],
    key: tuple[Any, ...],
    values: pd.Series,
    stride: int,
) -> None:
    store[key].update(values.to_numpy(), sample_stride=stride)


def _raw_feature_shift(
    config: dict[str, Any], output: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(config["physical_features_path"])
    features = config["raw_shift_features"]
    stride = int(config.get("quantile_sample_stride", 20))
    usecols = ["region", "year", "month", *features]
    overall: dict[tuple[Any, ...], RunningStats] = defaultdict(RunningStats)
    regional: dict[tuple[Any, ...], RunningStats] = defaultdict(RunningStats)
    monthly: dict[tuple[Any, ...], RunningStats] = defaultdict(RunningStats)
    for chunk in pd.read_csv(
        path, usecols=usecols, chunksize=int(config["chunksize"])
    ):
        chunk["partition"] = _partition(chunk["year"])
        chunk = chunk[chunk["partition"].isin(["train", "test"])]
        for partition, subset in chunk.groupby("partition", observed=True):
            for feature in features:
                _update_raw_stats(
                    overall, (partition, feature), subset[feature], stride
                )
            for region, group in subset.groupby("region", observed=True):
                for feature in features:
                    _update_raw_stats(
                        regional,
                        (partition, region, feature),
                        group[feature],
                        stride,
                    )
            for month, group in subset.groupby("month", observed=True):
                for feature in features:
                    _update_raw_stats(
                        monthly,
                        (partition, int(month), feature),
                        group[feature],
                        stride,
                    )

    outside_overall: dict[str, list[int]] = {
        feature: [0, 0] for feature in features
    }
    outside_region: dict[tuple[str, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    outside_month: dict[tuple[int, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for chunk in pd.read_csv(
        path, usecols=usecols, chunksize=int(config["chunksize"])
    ):
        test = chunk[chunk["year"] == 2023]
        for feature in features:
            values = test[feature].to_numpy()
            train_stats = overall[("train", feature)]
            outside_overall[feature][0] += int(
                ((values < train_stats.minimum) | (values > train_stats.maximum)).sum()
            )
            outside_overall[feature][1] += len(values)
        for region, group in test.groupby("region", observed=True):
            for feature in features:
                values = group[feature].to_numpy()
                stats = regional[("train", region, feature)]
                outside_region[(region, feature)][0] += int(
                    ((values < stats.minimum) | (values > stats.maximum)).sum()
                )
                outside_region[(region, feature)][1] += len(values)
        for month, group in test.groupby("month", observed=True):
            for feature in features:
                values = group[feature].to_numpy()
                stats = monthly[("train", int(month), feature)]
                outside_month[(int(month), feature)][0] += int(
                    ((values < stats.minimum) | (values > stats.maximum)).sum()
                )
                outside_month[(int(month), feature)][1] += len(values)

    summary_rows = []
    for feature in features:
        outside, count = outside_overall[feature]
        summary_rows.append(
            shift_row(
                feature,
                overall[("train", feature)].summary(),
                overall[("test", feature)].summary(),
                outside_ratio=outside / count,
                extra={"source": "physical_feature_csv"},
            )
        )
    region_rows = []
    for region in ["East China", "Sahara"]:
        for feature in features:
            outside, count = outside_region[(region, feature)]
            region_rows.append(
                shift_row(
                    feature,
                    regional[("train", region, feature)].summary(),
                    regional[("test", region, feature)].summary(),
                    outside_ratio=outside / count,
                    extra={"region": region},
                )
            )
    month_rows = []
    for month in range(1, 13):
        for feature in features:
            outside, count = outside_month[(month, feature)]
            month_rows.append(
                shift_row(
                    feature,
                    monthly[("train", month, feature)].summary(),
                    monthly[("test", month, feature)].summary(),
                    outside_ratio=outside / count,
                    extra={"month": month},
                )
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(region_rows),
        pd.DataFrame(month_rows),
    )


def _raw_feature_by_year(config: dict[str, Any]) -> pd.DataFrame:
    """Stream exact yearly feature moments without loading the full CSV."""
    path = Path(config["physical_features_path"])
    features = config["raw_shift_features"]
    store: dict[tuple[int, str, str], RunningStats] = defaultdict(RunningStats)
    usecols = ["region", "year", *features]
    for chunk in pd.read_csv(
        path, usecols=usecols, chunksize=int(config["chunksize"])
    ):
        for (year, region), subset in chunk.groupby(
            ["year", "region"], observed=True
        ):
            for feature in features:
                store[(int(year), str(region), feature)].update(
                    subset[feature].to_numpy(), sample_stride=20
                )
    rows = []
    for (year, region, feature), stats in sorted(store.items()):
        summary = stats.summary()
        rows.append(
            {
                "year": year,
                "region": region,
                "feature": feature,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _prepared_cache_diagnostics(
    config: dict[str, Any], output: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    cache = Path(config["prepared_temporal_cache"])
    samples = pd.read_pickle(cache)
    samples["partition"] = _partition(samples["target_year"])
    partition_rows: dict[str, Any] = {}
    for partition in ["train", "validation", "test"]:
        subset = samples[samples["partition"] == partition]
        partition_rows[partition] = {
            "sample_count": int(len(subset)),
            "years": sorted(int(value) for value in subset["target_year"].unique()),
            "region_sample_counts": {
                str(key): int(value)
                for key, value in subset["region"].value_counts().items()
            },
            "target_summary": _stats_summary(subset["y_true"]),
        }
    split_diagnostics = {
        "train_years": config["train_years"],
        "validation_years": config["validation_years"],
        "test_years": config["test_years"],
        "partitions": partition_rows,
        "prepared_cache": {
            "path": str(cache),
            "size_bytes": cache.stat().st_size,
        },
        "note": (
            "Temporal test contains only 2023, so per-year prediction "
            "diagnostics cannot distinguish multiple held-out years."
        ),
    }
    write_json(output / "split_diagnostics.json", split_diagnostics)

    dynamic_columns = [
        column
        for column in samples.columns
        if "_lag_" in column
        and not column.startswith("evaporation_anomaly")
    ]
    train = samples[samples["partition"] == "train"]
    test = samples[samples["partition"] == "test"]
    rows = []
    for column in dynamic_columns:
        train_values = train[column].to_numpy(dtype=np.float64)
        test_values = test[column].to_numpy(dtype=np.float64)
        train_summary = {
            "count": len(train_values),
            "mean": float(train_values.mean()),
            "std": float(train_values.std()),
            "min": float(train_values.min()),
            "p10": float(np.quantile(train_values, 0.10)),
            "p50": float(np.quantile(train_values, 0.50)),
            "p90": float(np.quantile(train_values, 0.90)),
            "max": float(train_values.max()),
        }
        test_summary = {
            "count": len(test_values),
            "mean": float(test_values.mean()),
            "std": float(test_values.std()),
            "min": float(test_values.min()),
            "p10": float(np.quantile(test_values, 0.10)),
            "p50": float(np.quantile(test_values, 0.50)),
            "p90": float(np.quantile(test_values, 0.90)),
            "max": float(test_values.max()),
        }
        outside = np.mean(
            (test_values < train_summary["min"])
            | (test_values > train_summary["max"])
        )
        rows.append(
            shift_row(
                column,
                train_summary,
                test_summary,
                outside_ratio=float(outside),
                extra={"source": "train_standardized_prepared_cache"},
            )
        )
    del train, test, samples
    return split_diagnostics, pd.DataFrame(rows)


def _write_behavior_notes(output: Path) -> None:
    seasonal = pd.read_csv(output / "per_season_metrics.csv")
    regional = pd.read_csv(output / "per_region_year_metrics.csv")
    residual = json.loads(
        (output / "residual_distribution_summary.json").read_text()
    )["rows"]
    shifts = pd.read_csv(output / "feature_shift_summary.csv")
    top_shift = shifts.reindex(
        shifts["standardized_mean_difference"].abs().sort_values(
            ascending=False
        ).index
    ).head(12)
    lines = [
        "# LightGBM model behavior notes",
        "",
        "- No fitted model or feature-importance artifact exists in the run "
        "directories. The runner saved metrics and predictions only; no model "
        "was retrained for this diagnosis.",
        "- Conclusions below are proxy diagnostics based on residuals, output "
        "variance, region/season decomposition, and train/test feature shift.",
        "",
        "## Largest standardized feature shifts",
        "",
        _markdown_table(
            top_shift[
                [
                    "feature",
                    "source",
                    "standardized_mean_difference",
                    "test_outside_train_range_ratio",
                ]
            ]
        ),
        "",
        "## Seasonal metrics",
        "",
        _markdown_table(
            seasonal[
                ["seed", "model_name", "season", "rmse", "bias", "r2"]
            ]
        ),
        "",
        "## Regional metrics",
        "",
        _markdown_table(
            regional[
                ["seed", "model_name", "region", "rmse", "bias", "r2"]
            ]
        ),
        "",
        "## Prediction-distribution proxy",
        "",
        _markdown_table(
            pd.DataFrame(residual)[
                [
                    "seed",
                    "model_name",
                    "bias",
                    "y_true_std",
                    "y_pred_std",
                    "prediction_to_target_std_ratio",
                    "abs_residual_p99",
                ]
            ]
        ),
    ]
    (output / "model_behavior_notes.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_linear_comparison(output: Path) -> None:
    residual = pd.DataFrame(
        json.loads(
            (output / "residual_distribution_summary.json").read_text()
        )["rows"]
    )
    aggregate = (
        residual.groupby("model_name")
        .agg(
            bias_mean=("bias", "mean"),
            rmse_mean=("rmse", "mean"),
            abs_residual_p99_mean=("abs_residual_p99", "mean"),
            y_true_std_mean=("y_true_std", "mean"),
            y_pred_std_mean=("y_pred_std", "mean"),
            prediction_to_target_std_ratio_mean=(
                "prediction_to_target_std_ratio",
                "mean",
            ),
        )
        .reset_index()
    )
    text = f"""# Linear versus LightGBM under temporal holdout

The temporal partition is fixed (train 2019–2021, validation 2022, test
2023). Linear is deterministic here, while the three LightGBM seeds measure
model stochasticity on the same temporal split.

{_markdown_table(aggregate)}

Interpretation must combine RMSE with R² and prediction variance. A lower
absolute RMSE across splits does not by itself establish easier
generalisation because the target variance differs between random and
temporal partitions. Region, season, residual-tail, and feature-shift tables
in this directory provide the supporting evidence.
"""
    (output / "linear_vs_lightgbm_temporal.md").write_text(
        text, encoding="utf-8"
    )


def _write_spatial_plan(config: dict[str, Any], output: Path) -> None:
    plan = config["repeated_spatial_plan"]
    validate_repeated_spatial_plan(plan)
    folds = int(plan["folds"])
    samples = 4_786_344
    test_rows = int(round(samples / folds))
    tasks = folds * len(plan["models"]) * len(plan["feature_sets"])
    cache_bytes = folds * 1_180_000_000
    full_prediction_bytes = tasks * test_rows * 85
    text = f"""# Repeated spatial folds plan

## Split construction

1. Build the spatial group key from `(region, latitude, longitude)` and assign
   each grid cell to exactly one 5° block.
2. Stratify blocks **within each region** before assigning them to {folds}
   folds. This avoids a seed accidentally changing the Sahara/East China
   composition as strongly as the current globally pooled block split.
3. Rotate one fold as test and one different fold as validation; all remaining
   folds are training. No grid cell or spatial block may appear in more than
   one partition within a fold.
4. Fit climatology, anomaly transforms, event thresholds, and standardisation
   independently on each fold's training rows.
5. Persist block-to-fold assignments and validate zero `grid_id` and block
   overlap before training.

## Reporting

- Report per-fold MAE, RMSE, R², Skill Score and region-level metrics.
- Report mean, sample standard deviation, min/max and all individual folds.
- Treat the fold, not individual grid rows, as the unit of spatial
  uncertainty.
- Include the held-out block count and region composition for every fold.

## Minimum matrix

- Models: Linear and LightGBM.
- Feature set: full only.
- Folds: {folds}.
- Total tasks: {tasks}.

LightGBM-only would require {folds} tasks, but retaining Linear gives a direct
robustness baseline for {tasks} tasks and is recommended.

## Estimated cost

- Approximate test rows per fold: {test_rows:,}.
- Prepared split cache: about {cache_bytes / 1e9:.1f} GB.
- Full canonical predictions: about {full_prediction_bytes / 1e9:.1f} GB.
- Recommended policy: metrics plus a deterministic 1% prediction sample for
  diagnostics, while recording the sampling rule in metadata.
- Expected wall time on the current machine: roughly 25–40 minutes.

No repeated-spatial experiment was run in this diagnostic stage because the
current split generator does not yet implement region-stratified rotating
folds; an ad-hoc smoke would not validate the proposed protocol.
"""
    (output / "repeated_spatial_folds_plan.md").write_text(
        text, encoding="utf-8"
    )


def _write_diagnostic_summary(output: Path) -> None:
    split = json.loads((output / "split_diagnostics.json").read_text())
    residual = pd.DataFrame(
        json.loads(
            (output / "residual_distribution_summary.json").read_text()
        )["rows"]
    )
    seasonal = pd.read_csv(output / "per_season_metrics.csv")
    regional = pd.read_csv(output / "per_region_year_metrics.csv")
    shifts = pd.read_csv(output / "feature_shift_summary.csv")
    yearly = pd.read_csv(output / "feature_shift_by_year.csv")

    lgb = residual[residual["model_name"] == "lightgbm"]
    linear = residual[residual["model_name"] == "linear_regression"]
    lgb_season = (
        seasonal[seasonal["model_name"] == "lightgbm"]
        .groupby("season")[["rmse", "bias"]]
        .mean()
        .sort_values("rmse", ascending=False)
    )
    lgb_region = (
        regional[regional["model_name"] == "lightgbm"]
        .groupby("region")[["rmse", "bias", "r2"]]
        .mean()
    )
    top = shifts.assign(
        absolute_smd=shifts["standardized_mean_difference"].abs()
    ).sort_values("absolute_smd", ascending=False).head(10)
    radiation_yearly = yearly[yearly["feature"] == "radiation"][
        ["year", "region", "mean"]
    ].pivot(index="year", columns="region", values="mean").reset_index()
    test = split["partitions"]["test"]
    text = f"""# ERA5-Land v1 temporal failure diagnosis

## Scope and split

- Train: 2019–2021 ({split["partitions"]["train"]["sample_count"]:,} samples).
- Validation: 2022 ({split["partitions"]["validation"]["sample_count"]:,} samples).
- Test: 2023 ({test["sample_count"]:,} samples).
- Test regions: {test["region_sample_counts"]}.
- The holdout contains one test year only. Failure can be localized to 2023,
  but not compared among multiple unseen test years without a rolling-origin
  experiment.

## Main finding

LightGBM's temporal failure is dominated by systematic positive bias, not
just random tail error. Across seeds its mean RMSE is {lgb["rmse"].mean():.3f},
mean bias is {lgb["bias"].mean():+.3f}, and it overpredicts
{100 * lgb["overprediction_ratio"].mean():.1f}% of test samples. The test
target mean is {test["target_summary"]["mean"]:.3f}, while LightGBM's mean
prediction is {lgb["y_pred_mean"].mean():.3f}. Its p99 absolute residual is
{lgb["abs_residual_p99"].mean():.3f}.

Linear is more conservative: RMSE {linear["rmse"].mean():.3f}, bias
{linear["bias"].mean():+.3f}, prediction/target standard-deviation ratio
{linear["prediction_to_target_std_ratio"].mean():.3f}, versus
{lgb["prediction_to_target_std_ratio"].mean():.3f} for LightGBM. Linear
under-disperses, but it follows the negative target-level shift much better.

## Season and region

{_markdown_table(lgb_season.reset_index())}

JJA is the hardest season. Its cross-seed mean LightGBM RMSE is
{lgb_season.loc["JJA", "rmse"]:.3f} and bias is
{lgb_season.loc["JJA", "bias"]:+.3f}; the largest monthly failure is August.

{_markdown_table(lgb_region.reset_index())}

East China has the larger absolute RMSE and a target mean far below zero.
However, Sahara has the worse relative R² and the larger radiation shift.
Therefore East China's higher absolute error is only partly consistent with
feature drift; it is also affected by its larger target scale and stronger
negative 2023 anomaly. The evidence does not support attributing all regional
error to covariate shift alone.

## Feature shift

{_markdown_table(top[["feature", "source", "standardized_mean_difference", "test_outside_train_range_ratio"]])}

Radiation is the dominant shift: all six standardized radiation-anomaly lags
move strongly negative, and raw radiation is also substantially lower in
2023. Precipitation is lower and dryness is higher, but their standardized
shifts are smaller. Because radiation changes across every calendar month,
this should be audited against the original ERA5 files and conversion logic
before interpreting it as a physical climate signal.

{_markdown_table(radiation_yearly)}

East China radiation is broadly stable through 2022 and then falls in 2023;
Sahara already falls in 2022 and drops much further in 2023. That asymmetric
step pattern makes a source/aggregation audit especially important.

These diagnostics establish association, not causal feature attribution.
Completed runs contain no fitted LightGBM model or importance artifact, and
no model was retrained.

## Next evaluation

Use the region-stratified five-fold spatial-block protocol in
`repeated_spatial_folds_plan.md`. Before expanding models, run a rolling-origin
temporal check (for example, sequentially holding out 2021, 2022, and 2023)
with train-only preprocessing. That is the minimum test needed to distinguish
a unique 2023 data/physical regime from generic future-year degradation.
"""
    (output / "diagnostic_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/diagnostics/era5_land_v1_temporal_failure.yaml",
    )
    parser.add_argument(
        "--refresh-derived",
        action="store_true",
        help=(
            "Refresh prediction metrics, yearly shifts, reports, and the code "
            "snapshot in an existing diagnostic directory."
        ),
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(config["output_dir"])
    if args.refresh_derived:
        if not output.is_dir():
            raise FileNotFoundError(
                f"Diagnostic directory does not exist: {output}"
            )
        _prediction_diagnostics(config, output)
        _raw_feature_by_year(config).to_csv(
            output / "feature_shift_by_year.csv", index=False
        )
        _write_behavior_notes(output)
        _write_linear_comparison(output)
        _write_spatial_plan(config, output)
        _write_diagnostic_summary(output)
        _git_snapshot(output)
        print(output)
        return
    output.mkdir(parents=True, exist_ok=False)
    save_yaml(config, output / "diagnostics_config.yaml")
    write_json(
        output / "input_runs.json",
        {
            "run_dirs": config["run_dirs"],
            "physical_features_path": config["physical_features_path"],
            "prepared_temporal_cache": config["prepared_temporal_cache"],
        },
    )
    _git_snapshot(output)

    prediction_audit = _prediction_diagnostics(config, output)
    split_diagnostics, lag_shift = _prepared_cache_diagnostics(config, output)
    raw_summary, raw_region, raw_month = _raw_feature_shift(config, output)
    pd.concat([raw_summary, lag_shift], ignore_index=True).to_csv(
        output / "feature_shift_summary.csv", index=False
    )
    raw_region.to_csv(output / "feature_shift_by_region.csv", index=False)
    raw_month.to_csv(output / "feature_shift_by_month.csv", index=False)
    _raw_feature_by_year(config).to_csv(
        output / "feature_shift_by_year.csv", index=False
    )
    write_json(
        output / "diagnostics_input_audit.json",
        {
            **prediction_audit,
            "physical_features_size_bytes": Path(
                config["physical_features_path"]
            ).stat().st_size,
            "prepared_cache_size_bytes": Path(
                config["prepared_temporal_cache"]
            ).stat().st_size,
            "model_artifact_available": False,
            "model_artifact_note": (
                "No serialized model or feature-importance artifact was "
                "saved; no retraining was performed."
            ),
            "split": split_diagnostics,
        },
    )
    (output / "lightgbm_feature_importance.csv.unavailable.txt").write_text(
        "Unavailable: completed runs contain no serialized fitted model or "
        "feature-importance artifact. No model was retrained.\n",
        encoding="utf-8",
    )
    _write_behavior_notes(output)
    _write_linear_comparison(output)
    _write_spatial_plan(config, output)
    _write_diagnostic_summary(output)
    print(output)


if __name__ == "__main__":
    main()
