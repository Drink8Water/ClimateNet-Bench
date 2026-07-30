"""Reproducible benchmark runner for ClimateNet-Bench.

Orchestrates: config → dataset → splits → models → train → evaluate → save.
"""

from __future__ import annotations

import hashlib
import gc
import importlib.metadata
import json
import logging
import platform
import re
import subprocess
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.benchmark.leaderboard import build_leaderboard
from climatenet.benchmark.manifest_split_adapter import (
    PROTOCOL as REPEATED_SPATIAL_PROTOCOL,
    load_manifest_backed_repeated_splits,
)
from climatenet.benchmark.split_protocols import (
    ALL_SPLIT_NAMES,
    SplitResult,
    generate_all_splits,
    validate_split,
)
from climatenet.data.forecasting_dataset import build_forecasting_samples
from climatenet.data.forecasting_dataset import (
    STATIC_FEATURE_COLUMNS,
    make_grid_id,
    make_sample_id,
)
from climatenet.data.loaders import load_csv
from climatenet.evaluation.metrics import evaluate_regression
from climatenet.models.model_factory import create_model
from climatenet.preprocessing.climatology import (
    TrainOnlyClimatePreprocessor,
    TrainOnlyStandardizer,
)
from climatenet.training.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
)
from climatenet.utils.config import save_yaml
from climatenet.utils.paths import ensure_directory, resolve_project_path
from climatenet.utils.random import set_random_seed

logger = logging.getLogger(__name__)

ANOMALY_SOURCE_COLUMNS: dict[str, str] = {
    "temperature_anomaly": "temperature",
    "precipitation_anomaly": "precipitation",
    "radiation_anomaly": "radiation",
    "soil_moisture_anomaly": "soil_moisture",
    "evaporation_anomaly": "evaporation",
}

SPLIT_PROTOCOL_NAME_MAP: dict[str, str] = {
    "random_split": "random",
    "spatial_block_holdout": "spatial_block",
    "temporal_holdout": "temporal",
    "spatial_temporal_holdout": "spatiotemporal",
    "region_transfer": "region_transfer",
    "climate_zone_transfer": "climate_zone_transfer",
    "repeated_region_stratified_spatial": (
        "repeated_region_stratified_spatial"
    ),
    # Internal names are also accepted for programmatic configs.
    **{name: name for name in ALL_SPLIT_NAMES},
}


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one input file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    """Hash the effective config independently of YAML formatting."""
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state() -> dict[str, Any]:
    """Return the current git revision and dirty flag without mutating state."""
    project_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def _data_source_identity(config: dict[str, Any]) -> tuple[str, bool, str]:
    """Return display name, synthetic flag, and filesystem-safe source slug."""
    data_source = str(config.get("data_source", "unknown")).strip() or "unknown"
    synthetic = bool(
        config.get("synthetic", "synthetic" in data_source.casefold())
    )
    slug = re.sub(r"[^a-z0-9]+", "-", data_source.casefold()).strip("-")
    return data_source, synthetic, slug or "unknown"


def _new_run_id(
    benchmark_name: str,
    source_slug: str,
    config_hash: str,
) -> tuple[str, str]:
    """Create a sortable, collision-resistant run ID and timestamp."""
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    benchmark_slug = re.sub(
        r"[^a-z0-9]+", "-", benchmark_name.casefold()
    ).strip("-") or "benchmark"
    run_id = (
        f"{benchmark_slug}-{source_slug}-{timestamp}-"
        f"{config_hash[:8]}-{uuid.uuid4().hex[:6]}"
    )
    return run_id, created_at


def _write_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)


def _package_environment() -> dict[str, Any]:
    """Return a concise, stable runtime environment snapshot."""
    packages: dict[str, str | None] = {}
    for distribution in [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
    ]:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _traceback_summary(exc: BaseException, max_lines: int = 20) -> str:
    """Return a bounded traceback suitable for a JSON audit artifact."""
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    flattened = "".join(lines).strip().splitlines()
    return "\n".join(flattened[-max_lines:])


def _finite_float(value: Any) -> float | None:
    """Convert an aggregate to JSON-safe float, replacing NaN/inf with null."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _data_audit(
    input_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Summarise the loaded data without changing it or fitting statistics."""
    warnings: list[str] = []
    if "target_year" in samples_df.columns:
        years = pd.to_numeric(samples_df["target_year"], errors="coerce")
        months = pd.to_numeric(
            samples_df.get("target_month", pd.Series(index=samples_df.index)),
            errors="coerce",
        )
    else:
        years = pd.to_numeric(input_df.get("year"), errors="coerce")
        months = pd.to_numeric(input_df.get("month"), errors="coerce")

    valid_dates = pd.DataFrame({"year": years, "month": months}).dropna()
    if valid_dates.empty:
        date_range = {"start": None, "end": None}
    else:
        date_values = (
            valid_dates["year"].astype(int).astype(str)
            + "-"
            + valid_dates["month"].astype(int).astype(str).str.zfill(2)
        )
        date_range = {"start": date_values.min(), "end": date_values.max()}

    target_series = (
        pd.to_numeric(samples_df[target_column], errors="coerce")
        if target_column in samples_df.columns
        else pd.Series(dtype=float)
    )
    target_finite = target_series[np.isfinite(target_series)]
    target_summary = {
        "scope": "loaded forecasting samples before split-specific preprocessing",
        "mean": _finite_float(target_finite.mean()),
        "std": _finite_float(target_finite.std()),
        "min": _finite_float(target_finite.min()),
        "max": _finite_float(target_finite.max()),
        "missing_count": int(target_series.isna().sum()),
        "non_finite_count": int((~np.isfinite(target_series)).sum()),
    }

    feature_summary: dict[str, Any] = {
        "checked_columns": [],
        "columns_with_non_finite": {},
        "total_non_finite": 0,
    }
    for column in dict.fromkeys(feature_columns):
        if column not in input_df.columns:
            continue
        values = pd.to_numeric(input_df[column], errors="coerce")
        count = int((~np.isfinite(values)).sum())
        feature_summary["checked_columns"].append(column)
        if count:
            feature_summary["columns_with_non_finite"][column] = count
            feature_summary["total_non_finite"] += count
    if feature_summary["total_non_finite"]:
        warnings.append(
            "Input features contain non-finite values; formal preprocessing "
            "will fail rather than impute them."
        )

    region_source = (
        input_df["region"] if "region" in input_df.columns
        else samples_df.get("region", pd.Series(dtype=object))
    )
    climate_column = next(
        (
            column
            for column in ["climate_type", "climate_zone"]
            if column in input_df.columns or column in samples_df.columns
        ),
        None,
    )
    climate_source = (
        input_df.get(climate_column, samples_df.get(climate_column))
        if climate_column is not None
        else None
    )
    audit = {
        "input_data_row_count": int(len(input_df)),
        "forecasting_sample_count": int(len(samples_df)),
        "date_range": date_range,
        "region_count": int(pd.Series(region_source).nunique(dropna=True)),
        "climate_zone_count": (
            int(pd.Series(climate_source).nunique(dropna=True))
            if climate_source is not None
            else None
        ),
        "target_summary": target_summary,
        "feature_non_finite_check": feature_summary,
    }
    return audit, warnings


def _aggregate_preprocessing_fallbacks(
    prepared_splits: dict[
        str, tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]
    ],
) -> tuple[dict[str, Any], list[str]]:
    """Aggregate split-level fallback counts into a run-level audit."""
    by_split: dict[str, Any] = {}
    totals = {
        "rows_checked": 0,
        "global_monthly_fallback_rows": 0,
        "global_mean_fallback_rows": 0,
        "fallback_samples": 0,
        "total_samples": 0,
    }
    for split_id, (_, _, metadata) in prepared_splits.items():
        usage = metadata.get("fallback_usage_by_partition", {})
        sample_counts = metadata.get("fallback_sample_counts", {})
        split_totals = {
            "rows_checked": 0,
            "global_monthly_fallback_rows": 0,
            "global_mean_fallback_rows": 0,
            "fallback_samples": 0,
            "total_samples": 0,
        }
        for partition_usage in usage.values():
            for variable_usage in partition_usage.values():
                split_totals["rows_checked"] += int(
                    variable_usage.get(
                        "rows", variable_usage.get("total_rows", 0)
                    )
                )
                split_totals["global_monthly_fallback_rows"] += int(
                    variable_usage.get("global_monthly_fallback_rows", 0)
                )
                split_totals["global_mean_fallback_rows"] += int(
                    variable_usage.get("global_mean_fallback_rows", 0)
                )
        for partition_counts in sample_counts.values():
            split_totals["fallback_samples"] += int(
                partition_counts.get("fallback_samples", 0)
            )
            split_totals["total_samples"] += int(
                partition_counts.get("total_samples", 0)
            )
        by_split[split_id] = {
            **split_totals,
            "by_partition_and_variable": usage,
            "sample_counts_by_partition": sample_counts,
        }
        for key in totals:
            totals[key] += split_totals[key]

    fallback_rows = (
        totals["global_monthly_fallback_rows"]
        + totals["global_mean_fallback_rows"]
    )
    ratio = fallback_rows / totals["rows_checked"] if totals["rows_checked"] else 0.0
    warnings: list[str] = []
    if ratio > 0.25:
        warnings.append(
            f"Preprocessing fallback ratio is high ({ratio:.1%} of transformed "
            "variable rows); review transfer-split coverage."
        )
    return {
        "totals": {**totals, "fallback_ratio": ratio},
        "by_split": by_split,
    }, warnings


def _build_run_summary(
    run_id: str,
    leaderboard_tables: dict[str, pd.DataFrame],
    completed: int,
    failed: int,
) -> dict[str, Any]:
    """Build a JSON-safe split and generalisation summary for one run."""
    all_results = leaderboard_tables.get("all_results", pd.DataFrame())
    split_performance: dict[str, Any] = {}
    if not all_results.empty:
        metric_columns = [
            column
            for column in ["rmse", "mae", "r2", "ood_degradation"]
            if column in all_results.columns
        ]
        for split_name, rows in all_results.groupby("split_protocol"):
            split_performance[str(split_name)] = {
                f"mean_{column}": _finite_float(rows[column].mean())
                for column in metric_columns
            } | {"task_count": int(len(rows))}

    ood_values = (
        pd.to_numeric(all_results["ood_degradation"], errors="coerce")
        if "ood_degradation" in all_results.columns
        else pd.Series(dtype=float)
    )
    generalisation = {
        "reference_split": "random",
        "mean_ood_degradation": _finite_float(ood_values.mean()),
        "definition": (
            "Relative RMSE change from random split for matching models; "
            "positive values indicate worse OOD performance."
        ),
    }
    if (
        not all_results.empty
        and "split_protocol" in all_results
        and not all_results["split_protocol"].eq("random").any()
    ):
        generalisation = {
            "reference_split": None,
            "mean_ood_degradation": None,
            "definition": (
                "Unavailable: this bounded run contains no random reference."
            ),
        }
    return {
        "run_id": run_id,
        "completed_task_count": completed,
        "failed_task_count": failed,
        "split_performance": split_performance,
        "generalisation": generalisation,
    }


def _repeated_fold_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated-spatial task metrics across audited folds."""
    if all_results.empty or "split_protocol" not in all_results:
        return pd.DataFrame()
    rows = all_results[
        all_results["split_protocol"].eq(REPEATED_SPATIAL_PROTOCOL)
    ]
    if rows.empty:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in [
            "mae",
            "rmse",
            "r2",
            "skill_vs_climatology",
            "zero_anomaly_climatology_rmse",
        ]
        if column in rows
    ]
    aggregation: dict[str, tuple[str, str]] = {}
    for column in metric_columns:
        aggregation[f"{column}_mean"] = (column, "mean")
        aggregation[f"{column}_std"] = (column, "std")
    return (
        rows.groupby(["model_name", "feature_set"], dropna=False)
        .agg(
            fold_count=("split_id", "nunique"),
            task_count=("experiment_id", "count"),
            **aggregation,
        )
        .reset_index()
    )


def _resolve_split_protocols(configured_names: list[str]) -> list[str]:
    """Map public config names to internal split protocol names."""
    unknown = [name for name in configured_names if name not in SPLIT_PROTOCOL_NAME_MAP]
    if unknown:
        raise ValueError(
            f"Unknown split protocol(s): {unknown}. "
            f"Supported config names: {sorted(SPLIT_PROTOCOL_NAME_MAP)}"
        )
    return [SPLIT_PROTOCOL_NAME_MAP[name] for name in configured_names]


def _parse_input_window(value: Any) -> int:
    """Return a positive input-window length from an int or e.g. ``'6 months'``."""
    if isinstance(value, bool):
        raise ValueError(f"input_window must be a positive integer, got {value!r}")
    if isinstance(value, int):
        window = value
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)(?:\s+months?)?\s*", value)
        if match is None:
            raise ValueError(
                f"input_window must be an integer or '<n> months', got {value!r}"
            )
        window = int(match.group(1))
    else:
        raise ValueError(f"input_window must be a positive integer, got {value!r}")
    if window <= 0:
        raise ValueError(f"input_window must be positive, got {window}")
    return window


def _expand_feature_names(requested: list[str], input_window: int) -> list[str]:
    """Expand dynamic feature names to their complete lag-column sequence."""
    static = set(STATIC_FEATURE_COLUMNS)
    expanded: list[str] = []
    for feature in requested:
        if feature in static:
            expanded.append(feature)
        else:
            expanded.extend(
                f"{feature}_lag_{lag}" for lag in range(1, input_window + 1)
            )
    return expanded


# ---------------------------------------------------------------------------
# Feature-set helpers
# ---------------------------------------------------------------------------


def _get_feature_columns(
    samples_df: pd.DataFrame,
    feature_set_name: str,
    feature_sets: dict,
    input_window: int,
) -> list[str]:
    """Resolve a configured feature set and require every resulting column."""
    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown feature_set={feature_set_name!r}")
    requested = feature_sets[feature_set_name].get("features", [])
    expanded = _expand_feature_names(requested, input_window)
    if not expanded:
        raise ValueError(
            f"Feature set {feature_set_name!r} does not request any features"
        )
    missing = [column for column in expanded if column not in samples_df.columns]
    if missing:
        raise ValueError(
            f"Feature set {feature_set_name!r} requires missing columns: {missing}"
        )
    return expanded


def _validate_benchmark_schema(
    samples_df: pd.DataFrame,
    feature_sets: dict,
    feature_set_names: list[str],
    input_window: int,
    target_column: str,
) -> dict[str, list[str]]:
    """Validate target, runner metadata, and every configured feature set."""
    required = [
        "sample_id",
        "grid_id",
        "region",
        "target_year",
        "target_month",
        target_column,
    ]
    missing = [column for column in required if column not in samples_df.columns]
    if missing:
        raise ValueError(f"Forecasting samples schema missing required columns: {missing}")

    return {
        name: _get_feature_columns(
            samples_df, name, feature_sets, input_window
        )
        for name in feature_set_names
    }


def _configured_dynamic_features(feature_sets: dict) -> list[str]:
    """Return unique non-static configured feature names in declaration order."""
    static = set(STATIC_FEATURE_COLUMNS)
    features: list[str] = []
    for feature_set in feature_sets.values():
        for feature in feature_set.get("features", []):
            if feature not in static and feature not in features:
                features.append(feature)
    return features


def _raw_source_column(feature: str, raw_df: pd.DataFrame) -> str:
    """Resolve a semantic anomaly name to its raw source column."""
    candidate = ANOMALY_SOURCE_COLUMNS.get(feature, feature)
    if candidate not in raw_df.columns:
        raise ValueError(
            f"Train-only preprocessing requires raw column {candidate!r} "
            f"for configured feature/target {feature!r}"
        )
    return candidate


def _add_forecast_sample_ids(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Attach IDs matching ``build_forecasting_samples`` target rows."""
    required = ["region", "year", "month", "latitude", "longitude"]
    missing = [column for column in required if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Raw feature table missing ID columns: {missing}")
    result = raw_df.copy()
    result["_forecast_sample_id"] = [
        make_sample_id(
            str(region),
            make_grid_id(float(latitude), float(longitude)),
            int(year),
            int(month),
        )
        for region, year, month, latitude, longitude in result[required].itertuples(
            index=False, name=None
        )
    ]
    return result


def _prepare_train_only_split(
    raw_features_df: pd.DataFrame,
    split_result: SplitResult,
    feature_sets: dict,
    input_window: int,
    target: str,
    target_column: str,
    standardize: bool = True,
    preprocessing_config: dict[str, Any] | None = None,
    artifact_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    """Fit on train, apply frozen transforms per partition, then build samples.

    A minimal raw forecasting table is created before this function solely to
    establish stable sample IDs for split generation. Model-ready samples are
    constructed here only after train/validation/test and history-context rows
    have been transformed with train-fitted statistics.
    """
    dynamic_features = _configured_dynamic_features(feature_sets)
    raw_with_ids = _add_forecast_sample_ids(raw_features_df)
    raw_with_ids["_raw_row_order"] = np.arange(len(raw_with_ids))
    train_mask = raw_with_ids["_forecast_sample_id"].isin(split_result.train_ids)
    val_mask = raw_with_ids["_forecast_sample_id"].isin(split_result.val_ids)
    test_mask = raw_with_ids["_forecast_sample_id"].isin(split_result.test_ids)
    context_mask = ~(train_mask | val_mask | test_mask)
    train_raw = raw_with_ids.loc[train_mask].drop(
        columns=["_forecast_sample_id", "_raw_row_order"]
    )
    if train_raw.empty:
        raise ValueError(
            f"Split {split_result.split_id!r} has no matching raw training rows"
        )

    anomaly_columns = {
        raw_col: semantic
        for semantic, raw_col in ANOMALY_SOURCE_COLUMNS.items()
        if raw_col in raw_features_df.columns
    }
    if target not in anomaly_columns.values():
        raw_target = _raw_source_column(target, raw_features_df)
        anomaly_columns[raw_target] = target

    climate_preprocessor = TrainOnlyClimatePreprocessor(anomaly_columns)
    climate_preprocessor.fit(train_raw)
    for partition, mask in [
        ("train", train_mask),
        ("validation", val_mask),
        ("test", test_mask),
        ("history_context", context_mask),
    ]:
        partition_df = raw_with_ids.loc[mask]
        if partition_df.empty:
            climate_preprocessor.fallback_usage.setdefault(partition, {})
            continue
        climate_preprocessor.record_fallback_usage(
            partition_df, partition=partition
        )
    transformed_raw = climate_preprocessor.transform(
        raw_with_ids.drop(columns=["_forecast_sample_id"]),
        partition="all",
        track_fallback=False,
    ).sort_values("_raw_row_order")
    transformed_raw = transformed_raw.drop(
        columns=["_raw_row_order"]
    ).reset_index(drop=True)

    for feature in dynamic_features:
        if feature not in transformed_raw.columns:
            # Non-anomaly dynamic features keep their raw name.
            _raw_source_column(feature, transformed_raw)

    samples_df, _ = build_forecasting_samples(
        transformed_raw,
        feature_columns=dynamic_features,
        target_column=target,
        sequence_length=input_window,
    )
    feature_set_names = list(feature_sets)
    resolved_feature_sets = _validate_benchmark_schema(
        samples_df,
        feature_sets,
        feature_set_names,
        input_window,
        target_column,
    )

    metadata = {
        "enabled": True,
        "workflow": "fit_train_then_transform_train_val_test",
        "source": "raw_feature_table",
        "climatology_group_keys": ["region", "month"],
        "training_years": sorted(int(year) for year in train_raw["year"].unique()),
        "training_regions": sorted(str(region) for region in train_raw["region"].unique()),
        "input_variables": list(anomaly_columns),
        "output_anomaly_variables": list(anomaly_columns.values()),
        "fallback_strategy": [
            "region_monthly",
            "global_monthly_fallback",
            "train_global_mean_for_month_absent_from_train",
        ],
        "validation_used_for_fit": False,
        "test_used_for_fit": False,
        "missing_value_handling": "error_no_imputation",
        "preprocessing_config": preprocessing_config or {},
        "preprocessing_config_hash": _config_hash(preprocessing_config or {}),
        "partition_calendar_months": {
            partition: sorted(
                int(month)
                for month in raw_with_ids.loc[mask, "month"].unique()
            )
            for partition, mask in [
                ("train", train_mask),
                ("validation", val_mask),
                ("test", test_mask),
            ]
        },
        "missing_partition_month_behavior": (
            "no_refit; absent rows require no action; a row for a month absent "
            "from train uses the train global mean fallback"
        ),
        **climate_preprocessor.metadata(),
    }

    fitted_pairs = set(
        zip(train_raw["region"].astype(str), train_raw["month"].astype(int))
    )
    fitted_months = set(train_raw["month"].astype(int))
    fallback_sample_counts: dict[str, dict[str, int]] = {}
    for partition, ids in [
        ("train", split_result.train_ids),
        ("validation", split_result.val_ids),
        ("test", split_result.test_ids),
    ]:
        partition_samples = samples_df[samples_df["sample_id"].isin(ids)]
        global_monthly_count = 0
        global_mean_count = 0
        for row in partition_samples[["region", "target_month"]].itertuples(
            index=False
        ):
            months = {
                ((int(row.target_month) - lag - 1) % 12) + 1
                for lag in range(0, input_window + 1)
            }
            missing_pairs = {
                month
                for month in months
                if (str(row.region), month) not in fitted_pairs
            }
            if missing_pairs:
                if any(month not in fitted_months for month in missing_pairs):
                    global_mean_count += 1
                else:
                    global_monthly_count += 1
        fallback_sample_counts[partition] = {
            "total_samples": int(len(partition_samples)),
            "global_monthly_fallback_samples": global_monthly_count,
            "global_mean_fallback_samples": global_mean_count,
            "fallback_samples": global_monthly_count + global_mean_count,
        }
    metadata["fallback_sample_counts"] = fallback_sample_counts

    if standardize:
        feature_columns = list(
            dict.fromkeys(
                column
                for columns in resolved_feature_sets.values()
                for column in columns
            )
        )
        train_samples = samples_df[
            samples_df["sample_id"].isin(split_result.train_ids)
        ]
        standardizer = TrainOnlyStandardizer(feature_columns).fit(train_samples)
        # Applying frozen train parameters to the whole table is a pure
        # transform. It avoids materialising and re-sorting four large copies.
        samples_df = standardizer.transform(samples_df)
        metadata.update(standardizer.metadata())
    else:
        metadata.update(
            {
                "standardization": "disabled",
                "standardization_fit_scope": "not_applicable",
            }
        )

    metadata["statistics_in_formal_runner"] = {
        "climatology": "train_only",
        "standardization": "train_only" if standardize else "disabled",
        "event_thresholds": "train_only",
        "missing_value_imputation": "not_connected; non-finite values error",
        "conformal_calibration": (
            "not_connected; conformal module requires independent validation "
            "calibration and must not use test"
        ),
    }
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata["climatology_artifacts"] = (
            climate_preprocessor.save_climatology_tables(artifact_dir)
        )
        _write_run_metadata(
            artifact_dir / "preprocessing_metadata.json",
            metadata,
        )

    return samples_df, resolved_feature_sets, metadata


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_benchmark(
    config: dict[str, Any],
    output_root: str | Path = "outputs/benchmark_runs",
) -> ExperimentRegistry:
    """Run one isolated benchmark and preserve failure metadata.

    Every invocation owns exactly one child directory of ``output_root``.
    Fatal setup errors are recorded in that directory before being re-raised.
    Per-task failures are handled by the implementation and do not abort the
    remaining task matrix.
    """
    output_base = Path(output_root)
    output_base.mkdir(parents=True, exist_ok=True)
    config_digest = _config_hash(config)
    benchmark_name = str(config.get("benchmark_name", "unnamed"))
    data_source, synthetic, source_slug = _data_source_identity(config)
    run_id, started_at = _new_run_id(
        benchmark_name, source_slug, config_digest
    )
    root = output_base / run_id
    root.mkdir(parents=True, exist_ok=False)
    git_state = _git_state()
    run_metadata: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        # Backward-compatible alias used by the leaderboard latest-run logic.
        "created_at": started_at,
        "finished_at": None,
        "status": "running",
        "benchmark_name": benchmark_name,
        "dataset": {
            "name": config.get("dataset_name", data_source),
            "version": config.get(
                "dataset_version", config.get("version", "unknown")
            ),
        },
        "data_source": data_source,
        "synthetic": synthetic,
        "config_hash": config_digest,
        "seed": int(config.get("random_seed", 42)),
        "target": config.get("target", "evaporation_anomaly"),
        "target_column": config.get("target_column", "y_true"),
        "git_commit": git_state["commit"],
        "git_dirty": git_state["dirty"],
        "environment": _package_environment(),
        "output_directory": str(root.resolve()),
        "data_files": {},
        "splits": [],
        "warnings": [],
        "total_models": 0,
        "total_splits": 0,
        "total_feature_sets": 0,
        "total_tasks": 0,
        "completed_task_count": 0,
        "failed_task_count": 0,
    }
    run_metadata_path = root / "run_metadata.json"
    run_metadata["artifact_policy"] = config.get("artifacts", {})
    run_metadata["quality_policy"] = config.get("quality_policy", {})
    run_metadata["audit_attachments"] = {}
    for audit_key in [
        "preflight_audit_path",
        "physical_features_audit_path",
        "code_snapshot_path",
        "fold_audit_path",
    ]:
        configured_path = config.get(audit_key)
        if not configured_path:
            continue
        audit_path = Path(configured_path)
        attachment: dict[str, Any] = {
            "path": str(audit_path.resolve()),
            "exists": audit_path.exists(),
        }
        if audit_path.is_file():
            attachment["sha256"] = _sha256_file(audit_path)
        elif audit_path.is_dir():
            attachment["files"] = sorted(
                str(path.relative_to(audit_path))
                for path in audit_path.rglob("*")
                if path.is_file()
            )
        run_metadata["audit_attachments"][audit_key] = attachment
    _write_run_metadata(run_metadata_path, run_metadata)

    try:
        return _run_benchmark_impl(
            config=config,
            root=root,
            run_metadata=run_metadata,
        )
    except Exception as exc:
        # Reload in case the implementation already persisted partial progress.
        try:
            persisted = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            persisted = run_metadata
        completed = int(persisted.get("completed_task_count", 0))
        persisted["status"] = "partial" if completed else "failed"
        persisted["finished_at"] = datetime.now(timezone.utc).isoformat()
        persisted["fatal_error"] = {
            "message": str(exc),
            "traceback_summary": _traceback_summary(exc),
        }
        _write_run_metadata(run_metadata_path, persisted)
        raise


def _run_benchmark_impl(
    config: dict[str, Any],
    root: Path,
    run_metadata: dict[str, Any],
) -> ExperimentRegistry:
    """Execute a benchmark inside an already-created isolated run directory."""
    seed = int(config.get("random_seed", 42))
    set_random_seed(seed)
    config_digest = str(run_metadata["config_hash"])
    data_source, synthetic, _ = _data_source_identity(config)
    run_id = str(run_metadata["run_id"])
    run_created_at = str(run_metadata["started_at"])
    registry = ExperimentRegistry(root / "experiment_registry.json")
    run_metadata_path = root / "run_metadata.json"

    benchmark_name = config.get("benchmark_name", "unnamed")
    models_cfg = config.get("models", [])
    configured_splits = config.get("split_protocols", ["random_split"])
    if not isinstance(configured_splits, list):
        raise ValueError("split_protocols must be a list")
    split_protocols = _resolve_split_protocols(configured_splits)
    feature_sets = config.get("feature_sets", {})
    input_window = _parse_input_window(config.get("input_window", 6))
    target_column = config.get("target_column", "y_true")
    preprocessing_config = config.get("preprocessing", {})
    train_only_preprocessing = bool(
        preprocessing_config.get("train_only", False)
    )
    selected_models = [
        model if isinstance(model, str) else model.get("name", "")
        for model in models_cfg
    ]
    resolved_config = {
        **config,
        "dataset": {
            "name": run_metadata["dataset"]["name"],
            "version": run_metadata["dataset"]["version"],
        },
        "target": config.get("target", "evaporation_anomaly"),
        "target_column": target_column,
        "split_protocols_configured": configured_splits,
        "split_protocols_resolved": split_protocols,
        "feature_sets_selected": list(feature_sets),
        "models_selected": selected_models,
        "random_seeds": [seed],
        "input_window": input_window,
        "preprocessing": preprocessing_config,
        "metrics": config.get("metrics", {}),
        "output_directory": str(root.resolve()),
        "git": {
            "commit": run_metadata["git_commit"],
            "dirty": run_metadata["git_dirty"],
        },
        "environment": run_metadata["environment"],
    }
    save_yaml(resolved_config, root / "config_resolved.yaml")

    for directory in [
        root / "splits",
        root / "preprocessing",
        root / "predictions",
        root / "metrics",
        root / "experiments",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("=== Benchmark: %s ===", benchmark_name)
    logger.info("Run: %s", run_id)
    logger.info("Data source: %s (synthetic=%s)", data_source, synthetic)
    logger.info("Models: %s", [m.get("name", m) for m in models_cfg])
    logger.info("Splits: %s", split_protocols)
    logger.info("Feature sets: %s", list(feature_sets.keys()))

    # ── Load or build forecasting dataset ──────────────────────────
    raw_features_df: pd.DataFrame | None = None
    data_paths: list[Path] = []
    audit_feature_columns: list[str] = []
    if train_only_preprocessing:
        features_path = config.get("features_path", "data/processed/features.csv")
        logger.info(
            "Building split basis from raw features for train-only preprocessing: %s",
            features_path,
        )
        resolved_features_path = resolve_project_path(features_path)
        data_paths.append(resolved_features_path)
        raw_features_df = load_csv(resolved_features_path)
        dynamic_features = _configured_dynamic_features(feature_sets)
        raw_dynamic_features = [
            _raw_source_column(feature, raw_features_df)
            for feature in dynamic_features
        ]
        audit_feature_columns = [
            column
            for column in [
                *STATIC_FEATURE_COLUMNS,
                *raw_dynamic_features,
            ]
            if column in raw_features_df.columns
        ]
        raw_target = _raw_source_column(
            config.get("target", "evaporation_anomaly"), raw_features_df
        )
        samples_df, _ = build_forecasting_samples(
            raw_features_df,
            # Split generation needs only stable identifiers and target dates.
            # Dynamic lag matrices are built after train-only preprocessing.
            feature_columns=[],
            target_column=raw_target,
            sequence_length=input_window,
        )
    else:
        samples_path = config.get(
            "forecasting_samples_path",
            "data/processed/forecasting_samples.csv",
        )
        samples_path = resolve_project_path(samples_path)
        if samples_path.exists():
            logger.info("Loading forecasting samples from %s", samples_path)
            data_paths.append(samples_path)
            samples_df = load_csv(samples_path)
        else:
            logger.info("Building forecasting samples from features …")
            features_path = config.get("features_path", "data/processed/features.csv")
            resolved_features_path = resolve_project_path(features_path)
            data_paths.append(resolved_features_path)
            features_df = load_csv(resolved_features_path)
            samples_df, _ = build_forecasting_samples(
                features_df,
                target_column=config.get("target", "evaporation_anomaly"),
                sequence_length=input_window,
            )
        audit_feature_columns = [
            column
            for feature_set in feature_sets.values()
            for column in _expand_feature_names(
                feature_set.get("features", []), input_window
            )
            if column in samples_df.columns
        ]

    logger.info("Samples: %d rows, %d grid cells",
                len(samples_df), samples_df["grid_id"].nunique())
    audit_input_df = (
        raw_features_df if raw_features_df is not None else samples_df
    )
    data_audit, audit_warnings = _data_audit(
        audit_input_df,
        samples_df,
        target_column,
        audit_feature_columns,
    )
    run_metadata.update(data_audit)
    run_metadata["warnings"].extend(audit_warnings)

    # Resolve and validate all requested features and the target before any
    # split/model experiment is started.
    feature_set_names = list(feature_sets.keys())
    if not feature_set_names:
        raise ValueError("At least one feature_set must be configured")
    resolved_feature_sets: dict[str, list[str]] | None = None
    if not train_only_preprocessing:
        resolved_feature_sets = _validate_benchmark_schema(
            samples_df,
            feature_sets,
            feature_set_names,
            input_window,
            target_column,
        )

    # ── Generate splits ────────────────────────────────────────────
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_configs = {
        "random": {"seed": seed},
        "spatial_block": {"seed": seed},
        "temporal": {"seed": seed},
        "region_transfer": {
            "seed": seed,
            "pairs": config.get("region_transfer_pairs"),
        },
        "climate_zone_transfer": {
            "seed": seed,
            "pairs": config.get("climate_zone_pairs"),
        },
        "spatiotemporal": {"seed": seed},
    }
    if REPEATED_SPATIAL_PROTOCOL in split_protocols:
        if split_protocols != [REPEATED_SPATIAL_PROTOCOL]:
            raise ValueError(
                "Manifest-backed repeated spatial must be the only protocol "
                "in a bounded run"
            )
        repeated_config = config.get("repeated_spatial")
        if not isinstance(repeated_config, dict):
            raise ValueError(
                "repeated_spatial configuration is required"
            )
        split_results = load_manifest_backed_repeated_splits(
            samples_df,
            repeated_config,
            splits_dir,
        )
        run_metadata["repeated_spatial_fold_audit"] = {
            "path": str(Path(repeated_config["audit_path"]).resolve()),
            "sha256": split_results[0].metadata["fold_audit_sha256"],
            "status": split_results[0].metadata["fold_audit_status"],
            "folds": [
                int(result.metadata["fold"]) for result in split_results
            ],
            "assignment_uses_target": False,
        }
    else:
        split_results = generate_all_splits(
            samples_df,
            splits_dir,
            configs=split_configs,
            protocols=split_protocols,
        )
    for split_result in split_results:
        split_errors = validate_split(samples_df, split_result)
        if split_errors:
            raise ValueError(
                f"Invalid split {split_result.split_id!r}: {split_errors}"
            )
    logger.info("Generated %d split(s)", len(split_results))

    use_split_cache = bool(config.get("memory_efficient_split_cache", False))
    prepared_splits: dict[
        str, tuple[pd.DataFrame | Path, dict[str, list[str]], dict[str, Any]]
    ] = {}
    for split_result in split_results:
        if train_only_preprocessing:
            assert raw_features_df is not None
            prepared = _prepare_train_only_split(
                raw_features_df,
                split_result,
                feature_sets,
                input_window,
                config.get("target", "evaporation_anomaly"),
                target_column,
                standardize=bool(
                    preprocessing_config.get("standardize_features", True)
                ),
                preprocessing_config=preprocessing_config,
                artifact_dir=root / "preprocessing" / split_result.split_id,
            )
            if use_split_cache:
                split_cache_path = (
                    root
                    / "preprocessing"
                    / split_result.split_id
                    / "prepared_samples.pkl"
                )
                prepared[0].to_pickle(split_cache_path)
                prepared[2]["prepared_sample_cache"] = {
                    "path": str(split_cache_path.relative_to(root)),
                    "purpose": (
                        "disk-backed split isolation to bound peak memory"
                    ),
                    "rows": int(len(prepared[0])),
                    "size_bytes": int(split_cache_path.stat().st_size),
                }
                _write_run_metadata(
                    root
                    / "preprocessing"
                    / split_result.split_id
                    / "preprocessing_metadata.json",
                    prepared[2],
                )
                prepared_splits[split_result.split_id] = (
                    split_cache_path,
                    prepared[1],
                    prepared[2],
                )
                del prepared
                gc.collect()
            else:
                prepared_splits[split_result.split_id] = prepared
        else:
            assert resolved_feature_sets is not None
            compatibility_metadata = {
                "enabled": False,
                "workflow": "precomputed_features_compatibility_mode",
                "validation_used_for_fit": "unknown_precomputed_input",
                "test_used_for_fit": "unknown_precomputed_input",
                "warning": (
                    "Not suitable for formal benchmark claims unless the "
                    "precomputed file was independently proven train-only."
                ),
            }
            prepared_splits[split_result.split_id] = (
                samples_df,
                resolved_feature_sets,
                compatibility_metadata,
            )
            _write_run_metadata(
                root
                / "preprocessing"
                / split_result.split_id
                / "preprocessing_metadata.json",
                compatibility_metadata,
            )

    run_metadata["data_files"] = {
        str(path.resolve()): _sha256_file(path)
        for path in dict.fromkeys(data_paths)
    }
    run_metadata["splits"] = [
        {
            "split_id": split_result.split_id,
            "protocol": split_result.protocol,
            "seed": int(split_result.config.get("seed", seed)),
            "n_train": len(split_result.train_ids),
            "n_validation": len(split_result.val_ids),
            "n_test": len(split_result.test_ids),
            "config": split_result.config,
            "feature_columns": prepared_splits[split_result.split_id][1],
        }
        for split_result in split_results
    ]
    fallback_summary, fallback_warnings = _aggregate_preprocessing_fallbacks(
        prepared_splits
    )
    run_metadata["preprocessing_fallback_summary"] = fallback_summary
    run_metadata["warnings"].extend(fallback_warnings)
    run_metadata["transformed_target_summary_by_split"] = {}
    for split_result in split_results:
        split_source = prepared_splits[split_result.split_id][0]
        split_samples = (
            pd.read_pickle(split_source)
            if isinstance(split_source, Path)
            else split_source
        )
        partition_summaries: dict[str, Any] = {}
        for partition, ids in [
            ("train", split_result.train_ids),
            ("validation", split_result.val_ids),
            ("test", split_result.test_ids),
        ]:
            values = pd.to_numeric(
                split_samples.loc[
                    split_samples["sample_id"].isin(ids), target_column
                ],
                errors="coerce",
            )
            finite_values = values[np.isfinite(values)]
            partition_summaries[partition] = {
                "row_count": int(len(values)),
                "mean": _finite_float(finite_values.mean()),
                "std": _finite_float(finite_values.std()),
                "min": _finite_float(finite_values.min()),
                "max": _finite_float(finite_values.max()),
                "missing_count": int(values.isna().sum()),
                "non_finite_count": int((~np.isfinite(values)).sum()),
            }
        run_metadata["transformed_target_summary_by_split"][
            split_result.split_id
        ] = partition_summaries
        if isinstance(split_source, Path):
            del split_samples
            gc.collect()
    for split_result in split_results:
        if min(len(split_result.train_ids), len(split_result.test_ids)) < 10:
            run_metadata["warnings"].append(
                f"Split {split_result.split_id!r} has fewer than 10 samples "
                "in train or test."
            )
    resolved_config["feature_columns_resolved"] = {
        split_id: feature_columns
        for split_id, (_, feature_columns, _) in prepared_splits.items()
    }
    resolved_config["data_files"] = run_metadata["data_files"]
    save_yaml(resolved_config, root / "config_resolved.yaml")
    _write_run_metadata(run_metadata_path, run_metadata)

    # ── Run experiments ────────────────────────────────────────────
    experiments_dir = root / "experiments"
    metrics_dir = root / "metrics"
    predictions_dir = root / "predictions"
    baseline_names = {"climatology", "persistence"}
    total_tasks = sum(
        len(split_results)
        * (1 if (
            (model if isinstance(model, str) else model.get("name", ""))
            in baseline_names
        ) else len(feature_set_names))
        for model in models_cfg
    )
    run_metadata.update(
        {
            "total_models": len(models_cfg),
            "total_splits": len(split_results),
            "total_feature_sets": len(feature_set_names),
            "total_tasks": total_tasks,
        }
    )
    _write_run_metadata(run_metadata_path, run_metadata)

    for model_cfg in models_cfg:
        if isinstance(model_cfg, str):
            model_name = model_cfg
            model_kwargs = {}
        else:
            model_name = model_cfg.get("name", "")
            model_kwargs: dict[str, Any] = {}
            for k, v in model_cfg.items():
                if k in ("name", "type", "description"):
                    continue
                if k == "params" and isinstance(v, dict):
                    model_kwargs.update(v)  # flatten params
                else:
                    model_kwargs[k] = v

        # Baselines don't need feature sets
        is_baseline = model_name in baseline_names

        for split_result in split_results:
            split_source, split_feature_sets, preprocessing_metadata = (
                prepared_splits[split_result.split_id]
            )
            split_samples_df = (
                pd.read_pickle(split_source)
                if isinstance(split_source, Path)
                else split_source
            )
            # Tree models do temporal splits too now
            for fs_name in feature_set_names:
                if is_baseline and fs_name != feature_set_names[0]:
                    continue  # Baselines only run once per split

                experiment_id = (
                    f"{benchmark_name}_{model_name}"
                    f"_{split_result.split_id}_{fs_name}_seed{seed}"
                )
                exp_dir = experiments_dir / experiment_id
                exp_dir.mkdir(parents=True, exist_ok=True)

                record = ExperimentRecord(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    benchmark_name=benchmark_name,
                    data_source=data_source,
                    synthetic=synthetic,
                    model_name=model_name,
                    split_protocol=split_result.protocol,
                    feature_set=fs_name,
                    train_regions=sorted(set(
                        split_samples_df[split_samples_df["sample_id"].isin(split_result.train_ids)]["region"]
                    )),
                    test_regions=sorted(set(
                        split_samples_df[split_samples_df["sample_id"].isin(split_result.test_ids)]["region"]
                    )),
                    train_years=sorted(set(
                        split_samples_df[split_samples_df["sample_id"].isin(split_result.train_ids)]["target_year"]
                    )),
                    test_years=sorted(set(
                        split_samples_df[split_samples_df["sample_id"].isin(split_result.test_ids)]["target_year"]
                    )),
                    preprocessing=preprocessing_metadata,
                    seed=seed,
                )
                registry.add(record)
                registry.save()

                try:
                    # A model instance is never shared across experiments.
                    experiment_model_kwargs = dict(model_kwargs)
                    if not is_baseline:
                        experiment_model_kwargs["random_state"] = seed
                    set_random_seed(seed)
                    model = create_model(model_name, experiment_model_kwargs)
                    resolved_model_name = model.get_model_name()
                    record.model_name = resolved_model_name
                    logger.info(
                        "Created fresh model: %s (%s / %s / seed=%d)",
                        resolved_model_name,
                        split_result.split_id,
                        fs_name,
                        seed,
                    )
                    registry.mark_running(experiment_id)
                    registry.save()
                    metrics, predictions = _run_one_experiment(
                        samples_df=split_samples_df,
                        split_result=split_result,
                        model=model,
                        model_name=resolved_model_name,
                        feature_set_name=fs_name,
                        feature_columns=split_feature_sets[fs_name],
                        is_baseline=is_baseline,
                        exp_dir=exp_dir,
                        config=config,
                        seed=seed,
                        run_metadata={
                            "run_id": run_id,
                            "run_created_at": run_created_at,
                            "data_source": data_source,
                            "synthetic": synthetic,
                            "config_hash": config_digest,
                            "seed": seed,
                            "target": config.get(
                                "target", "evaporation_anomaly"
                            ),
                            "target_column": target_column,
                            "data_file_hashes": run_metadata["data_files"],
                            "git_commit": run_metadata["git_commit"],
                            "git_dirty": run_metadata["git_dirty"],
                        },
                    )
                    canonical_metrics_path = metrics_dir / f"{experiment_id}.json"
                    canonical_predictions_path = (
                        predictions_dir / f"{experiment_id}.csv"
                    )
                    _write_run_metadata(canonical_metrics_path, metrics)
                    artifact_config = config.get("artifacts", {})
                    predictions.to_csv(
                        canonical_predictions_path,
                        index=False,
                        float_format=artifact_config.get(
                            "prediction_float_format"
                        ),
                    )
                    record.metrics_path = str(
                        canonical_metrics_path.relative_to(root)
                    )
                    record.predictions_path = str(
                        canonical_predictions_path.relative_to(root)
                    )
                    registry.mark_completed(experiment_id)
                    logger.info("✅ %s", experiment_id)
                except Exception as e:
                    logger.error("❌ %s: %s", experiment_id, e)
                    trace = _traceback_summary(e)
                    registry.mark_failed(experiment_id, str(e), trace)
                    failure = {
                        "status": "failed",
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "model_name": model_name,
                        "split_protocol": split_result.protocol,
                        "split_id": split_result.split_id,
                        "feature_set": fs_name,
                        "seed": seed,
                        "error_message": str(e),
                        "traceback_summary": trace,
                    }
                    failure_path = metrics_dir / f"{experiment_id}.json"
                    _write_run_metadata(failure_path, failure)
                    _write_run_metadata(exp_dir / "failure.json", failure)
                    record.metrics_path = str(failure_path.relative_to(root))
                finally:
                    registry.save()
                    run_metadata["completed_task_count"] = len(
                        registry.list_completed()
                    )
                    run_metadata["failed_task_count"] = len(
                        registry.list_failed()
                    )
                    _write_run_metadata(run_metadata_path, run_metadata)
            if isinstance(split_source, Path):
                del split_samples_df
                gc.collect()

    registry.save()
    completed_count = len(registry.list_completed())
    failed_count = len(registry.list_failed())
    if failed_count and completed_count:
        run_metadata["status"] = "partial"
    elif failed_count:
        run_metadata["status"] = "failed"
    else:
        run_metadata["status"] = "completed"
    run_metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    # Backward-compatible aliases.
    run_metadata["completed_at"] = run_metadata["finished_at"]
    run_metadata["n_completed"] = completed_count
    run_metadata["n_failed"] = failed_count
    run_metadata["completed_task_count"] = completed_count
    run_metadata["failed_task_count"] = failed_count
    leaderboard_tables = build_leaderboard(
        experiments_root=experiments_dir,
        output_root=root,
        run_id=run_id,
    ) if completed_count else {}
    summary = _build_run_summary(
        run_id,
        leaderboard_tables,
        completed_count,
        failed_count,
    )
    all_results = leaderboard_tables.get("all_results", pd.DataFrame())
    across_fold = _repeated_fold_summary(all_results)
    if not across_fold.empty:
        across_fold.to_csv(root / "across_fold_summary.csv", index=False)
        across_fold_payload = across_fold.to_dict(orient="records")
        _write_run_metadata(
            root / "across_fold_summary.json",
            {
                "run_id": run_id,
                "protocol": REPEATED_SPATIAL_PROTOCOL,
                "aggregation": "mean_std_across_folds",
                "rows": across_fold_payload,
            },
        )
        summary["across_fold_summary"] = across_fold_payload
    _write_run_metadata(root / "summary.json", summary)
    artifact_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path != run_metadata_path
    ]
    prediction_files = list((root / "predictions").glob("*.csv"))
    run_metadata["artifact_summary"] = {
        # Include run_metadata.json itself without making its recursive byte
        # size part of the checksum-like total.
        "file_count": len(artifact_files) + 1,
        "total_bytes_excluding_run_metadata": int(
            sum(path.stat().st_size for path in artifact_files)
        ),
        "prediction_file_count": len(prediction_files),
        "prediction_total_bytes": int(
            sum(path.stat().st_size for path in prediction_files)
        ),
        "compatibility_prediction_file_count": len(
            list((root / "experiments").glob("*/predictions.csv"))
        ),
    }
    _write_run_metadata(run_metadata_path, run_metadata)
    logger.info("Registry saved: %d completed, %d failed",
                completed_count, failed_count)
    return registry


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------


def _run_one_experiment(
    samples_df: pd.DataFrame,
    split_result: SplitResult,
    model: Any,
    model_name: str,
    feature_set_name: str,
    feature_columns: list[str],
    is_baseline: bool,
    exp_dir: Path,
    config: dict,
    seed: int,
    run_metadata: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train, predict, evaluate for one (model × split × feature_set)."""
    train_df = samples_df[samples_df["sample_id"].isin(split_result.train_ids)].copy()
    val_df = samples_df[samples_df["sample_id"].isin(split_result.val_ids)].copy()
    test_df = samples_df[samples_df["sample_id"].isin(split_result.test_ids)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Empty train or test set.")

    # ── features ─────────────────────────────────────────────────
    if is_baseline:
        feature_cols = []  # baselines ignore feature_columns
    else:
        feature_cols = feature_columns

    target_col = config.get("target_column", "y_true")

    # ── fit ───────────────────────────────────────────────────────
    fit_kwargs: dict = {"train_df": train_df, "target_column": target_col}
    if not is_baseline:
        fit_kwargs["feature_columns"] = feature_cols
    if val_df is not None and not val_df.empty and not is_baseline:
        fit_kwargs["val_df"] = val_df

    model.fit(**fit_kwargs)

    # ── predict ──────────────────────────────────────────────────
    y_pred = model.predict(test_df)
    y_true = test_df[target_col].to_numpy(dtype=np.float64)

    # ── primary metrics ──────────────────────────────────────────
    metrics = evaluate_regression(y_true, y_pred)
    zero_anomaly_rmse = float(np.sqrt(np.mean(np.square(y_true))))
    metrics["zero_anomaly_climatology_rmse"] = zero_anomaly_rmse
    metrics["skill_vs_climatology"] = (
        float(1.0 - metrics["rmse"] / zero_anomaly_rmse)
        if zero_anomaly_rmse > 0
        else float("nan")
    )
    metrics["skill_score"] = metrics["skill_vs_climatology"]
    metrics["model_name"] = model_name
    metrics["split_protocol"] = split_result.protocol
    metrics["feature_set"] = feature_set_name
    metrics["n_train"] = len(train_df)
    metrics["n_val"] = len(val_df)
    metrics["n_test"] = len(test_df)
    metrics.update(run_metadata)
    metrics["feature_columns"] = feature_cols
    metrics["split_id"] = split_result.split_id
    if "fold" in split_result.metadata:
        metrics["fold"] = int(split_result.metadata["fold"])
    metrics["regional_metrics"] = {}
    for region in sorted(test_df["region"].astype(str).unique()):
        region_mask = test_df["region"].astype(str).to_numpy() == region
        metrics["regional_metrics"][region] = {
            **evaluate_regression(y_true[region_mask], y_pred[region_mask]),
            "n_test": int(region_mask.sum()),
        }

    # Save metrics
    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ── predictions ──────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "sample_id": test_df["sample_id"].tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "partition": "test",
        "model_name": model_name,
        "split_protocol": split_result.protocol,
        "split_id": split_result.split_id,
        "feature_set": feature_set_name,
        "seed": seed,
    })
    if "fold" in split_result.metadata:
        pred_df["fold"] = int(split_result.metadata["fold"])
    # Merge metadata from test_df
    for col in ["region", "climate_type", "target_year", "target_month",
                "latitude", "longitude", "grid_id"]:
        if col in test_df.columns:
            pred_df[col] = test_df[col].to_numpy()
    artifact_config = config.get("artifacts", {})
    configured_columns = artifact_config.get("prediction_columns")
    if configured_columns:
        missing_prediction_columns = [
            column for column in configured_columns if column not in pred_df
        ]
        if missing_prediction_columns:
            raise ValueError(
                "Configured prediction artifact columns are unavailable: "
                f"{missing_prediction_columns}"
            )
        pred_df = pred_df[configured_columns]
    if artifact_config.get("compatibility_predictions", True):
        pred_df.to_csv(
            exp_dir / "predictions.csv",
            index=False,
            float_format=artifact_config.get("prediction_float_format"),
        )

    # ── config snapshot ───────────────────────────────────────────
    save_yaml(config, exp_dir / "config.yaml")
    return metrics, pred_df
