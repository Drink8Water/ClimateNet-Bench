"""Read-only diagnostics for seed-dependent spatial split composition."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PARTITIONS = ("train", "validation", "test")


def resolve_run_dirs(config: dict[str, Any]) -> dict[int, Path]:
    """Resolve and cross-check run paths against the authoritative summary."""
    summary_root = Path(config["multi_seed_summary_path"])
    summary_path = summary_root / config.get(
        "multi_seed_summary_file", "multiseed_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_dirs = {
        int(seed): Path(path) for seed, path in summary["run_dirs"].items()
    }
    expected_seeds = [int(seed) for seed in config["seeds"]]
    if set(summary_dirs) != set(expected_seeds):
        raise ValueError(
            "Summary seeds do not match diagnostic seeds: "
            f"{sorted(summary_dirs)} != {sorted(expected_seeds)}"
        )
    configured = {
        int(seed): Path(path)
        for seed, path in config.get("run_dirs", {}).items()
    }
    if configured and configured != summary_dirs:
        raise ValueError(
            "Configured run_dirs do not exactly match multiseed summary"
        )
    for seed, run_dir in summary_dirs.items():
        metadata_path = run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "completed":
            raise ValueError(f"Seed {seed} run is not completed: {run_dir}")
        if int(metadata.get("seed", -1)) != seed:
            raise ValueError(f"Seed mismatch for run {run_dir}")
    return summary_dirs


def _grid_key_from_sample_id(
    sample_ids: pd.Series, regions: Iterable[str]
) -> pd.DataFrame:
    """Extract region/grid identifiers, dropping the `_YYYY_MM` suffix."""
    ids = sample_ids.astype("string")
    rows: list[pd.DataFrame] = []
    matched = pd.Series(False, index=ids.index)
    for region in regions:
        prefix = f"{region}_"
        mask = ids.str.startswith(prefix, na=False)
        if not mask.any():
            continue
        grid_id = ids.loc[mask].str[len(prefix) :].str.rsplit("_", n=2).str[0]
        rows.append(
            pd.DataFrame({"region": region, "grid_id": grid_id.to_numpy()})
        )
        matched.loc[mask] = True
    if not bool(matched.all()):
        examples = ids.loc[~matched].head(3).tolist()
        raise ValueError(f"Cannot parse region from sample IDs: {examples}")
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def load_spatial_grid_assignments(
    split_dir: Path, regions: Iterable[str]
) -> pd.DataFrame:
    """Load unique spatial grid assignments without retaining millions of IDs."""
    assignments: list[pd.DataFrame] = []
    for partition in PARTITIONS:
        path = split_dir / f"{partition if partition != 'validation' else 'val'}_ids.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        pieces: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, usecols=["sample_id"], chunksize=250_000):
            pieces.append(_grid_key_from_sample_id(chunk["sample_id"], regions))
        grids = pd.concat(pieces, ignore_index=True).drop_duplicates()
        grids["partition"] = partition
        assignments.append(grids)
    result = pd.concat(assignments, ignore_index=True)
    duplicate = result.duplicated(["region", "grid_id"], keep=False)
    if duplicate.any():
        raise ValueError("A spatial grid cell appears in multiple partitions")
    return result


def assign_partitions(
    samples: pd.DataFrame, assignments: pd.DataFrame
) -> pd.Series:
    """Map each prepared sample to a spatial partition by region and grid."""
    partition = pd.Series(index=samples.index, dtype="string")
    for region, mapping_rows in assignments.groupby("region", observed=True):
        mapping = mapping_rows.set_index("grid_id")["partition"].to_dict()
        mask = samples["region"].astype("string").eq(str(region))
        partition.loc[mask] = (
            samples.loc[mask, "grid_id"].astype("string").map(mapping)
        )
    if partition.isna().any():
        missing = samples.loc[
            partition.isna(), ["region", "grid_id", "sample_id"]
        ].head(3)
        raise ValueError(
            "Prepared samples are missing spatial assignments: "
            f"{missing.to_dict(orient='records')}"
        )
    return partition


def target_summary(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Target summary requires finite observations")
    return {
        "sample_count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "zero_anomaly_baseline_rmse": float(
            np.sqrt(np.mean(np.square(array)))
        ),
    }


def target_distribution_rows(
    samples: pd.DataFrame, seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition, group in samples.groupby("partition", observed=True):
        rows.append(
            {
                "seed": seed,
                "partition": partition,
                "region": "ALL",
                **target_summary(group["y_true"]),
            }
        )
        for region, regional in group.groupby("region", observed=True):
            rows.append(
                {
                    "seed": seed,
                    "partition": partition,
                    "region": str(region),
                    **target_summary(regional["y_true"]),
                }
            )
    return rows


def _rmse(y_true: np.ndarray, y_pred: np.ndarray | float) -> float:
    return float(
        np.sqrt(np.mean(np.square(y_true - np.asarray(y_pred))))
    )


def naive_baseline_rows(
    samples: pd.DataFrame,
    seed: int,
    standardization_parameters: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Compute train-fitted or parameter-free baselines without model fitting."""
    train = samples[samples["partition"] == "train"]
    test = samples[samples["partition"] == "test"].copy()
    train_mean = float(train["y_true"].mean())
    region_month = train.groupby(
        ["region", "target_month"], observed=True
    )["y_true"].mean()
    keys = pd.MultiIndex.from_frame(test[["region", "target_month"]])
    region_month_prediction = region_month.reindex(keys).to_numpy()
    if not np.isfinite(region_month_prediction).all():
        raise ValueError("Region-month target baseline has missing train groups")

    predictors: dict[str, np.ndarray] = {
        "zero_anomaly": np.zeros(len(test), dtype=np.float64),
        "train_target_mean": np.full(len(test), train_mean),
        "train_region_month_target_mean": region_month_prediction,
    }
    persistence_column = "evaporation_anomaly_lag_1"
    params = standardization_parameters.get(persistence_column)
    if persistence_column in test:
        persistence = test[persistence_column].to_numpy(dtype=np.float64)
        # This auxiliary lag is currently retained in physical anomaly units
        # and is not one of the standardized model inputs. Support both that
        # representation and any future cache that records scaler parameters.
        if params:
            persistence = (
                persistence * float(params["scale"]) + float(params["mean"])
            )
        predictors["persistence_lag_1"] = persistence

    rows: list[dict[str, Any]] = []
    for region in ["ALL", *sorted(test["region"].astype(str).unique())]:
        mask = (
            np.ones(len(test), dtype=bool)
            if region == "ALL"
            else test["region"].astype(str).to_numpy() == region
        )
        truth = test["y_true"].to_numpy(dtype=np.float64)[mask]
        for name, prediction in predictors.items():
            rows.append(
                {
                    "seed": seed,
                    "region": region,
                    "baseline": name,
                    "sample_count": int(mask.sum()),
                    "rmse": _rmse(truth, prediction[mask]),
                }
            )
    return rows


def resolve_feature_columns(
    columns: Iterable[str], patterns: Iterable[str]
) -> list[str]:
    resolved = sorted(
        {
            column
            for pattern in patterns
            for column in columns
            if fnmatch.fnmatch(column, pattern)
        }
    )
    if not resolved:
        raise ValueError("No prepared feature columns match configured patterns")
    return resolved


def feature_shift_row(
    train_values: np.ndarray,
    test_values: np.ndarray,
    *,
    seed: int,
    feature: str,
    quantile_stride: int = 20,
) -> dict[str, Any]:
    train = np.asarray(train_values, dtype=np.float64)
    test = np.asarray(test_values, dtype=np.float64)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if not train.size or not test.size:
        raise ValueError(f"Feature {feature} has no finite train/test values")
    train_mean, test_mean = float(train.mean()), float(test.mean())
    train_std, test_std = float(train.std()), float(test.std())
    pooled = float(np.sqrt((train_std**2 + test_std**2) / 2))
    train_sample = train[::quantile_stride]
    test_sample = test[::quantile_stride]
    train_min, train_max = float(train.min()), float(train.max())
    return {
        "seed": seed,
        "feature": feature,
        "train_count": int(train.size),
        "test_count": int(test.size),
        "train_mean": train_mean,
        "test_mean": test_mean,
        "train_std": train_std,
        "test_std": test_std,
        "standardized_mean_difference": (
            (test_mean - train_mean) / pooled if pooled else 0.0
        ),
        "train_p10": float(np.quantile(train_sample, 0.10)),
        "test_p10": float(np.quantile(test_sample, 0.10)),
        "train_p50": float(np.quantile(train_sample, 0.50)),
        "test_p50": float(np.quantile(test_sample, 0.50)),
        "train_p90": float(np.quantile(train_sample, 0.90)),
        "test_p90": float(np.quantile(test_sample, 0.90)),
        "p10_shift": float(
            np.quantile(test_sample, 0.10)
            - np.quantile(train_sample, 0.10)
        ),
        "p50_shift": float(
            np.quantile(test_sample, 0.50)
            - np.quantile(train_sample, 0.50)
        ),
        "p90_shift": float(
            np.quantile(test_sample, 0.90)
            - np.quantile(train_sample, 0.90)
        ),
        "test_outside_train_range_ratio": float(
            np.mean((test < train_min) | (test > train_max))
        ),
    }


def composition_tables(
    samples: pd.DataFrame, seed: int
) -> dict[str, pd.DataFrame]:
    partition = (
        samples.groupby("partition", observed=True)
        .size()
        .rename("sample_count")
        .reset_index()
    )
    partition.insert(0, "seed", seed)
    partition["sample_share"] = partition["sample_count"] / len(samples)

    region = (
        samples.groupby(["partition", "region"], observed=True)
        .size()
        .rename("sample_count")
        .reset_index()
    )
    totals = region.groupby("partition")["sample_count"].transform("sum")
    region.insert(0, "seed", seed)
    region["partition_share"] = region["sample_count"] / totals

    grid = (
        samples.groupby("partition", observed=True)
        .apply(
            lambda frame: frame[["region", "grid_id"]]
            .drop_duplicates()
            .shape[0],
            include_groups=False,
        )
        .rename("unique_grid_cell_count")
        .reset_index()
    )
    grid.insert(0, "seed", seed)

    month = (
        samples.groupby(["partition", "target_month"], observed=True)
        .size()
        .rename("sample_count")
        .reset_index()
    )
    month.insert(0, "seed", seed)
    coverage = (
        samples.groupby("partition", observed=True)["target_month"]
        .nunique()
        .rename("covered_month_count")
    )
    month = month.join(coverage, on="partition")
    return {
        "partition": partition,
        "region": region,
        "grid": grid,
        "month": month,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])
