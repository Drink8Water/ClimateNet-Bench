"""Guarded preparation and execution of a small real ERA5-Land dry-run."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.data.era5_audit import audit_era5_files
from climatenet.data.era5_preprocess import preprocess_era5_file
from climatenet.features.physical import add_physical_features
from climatenet.training.benchmark_runner import run_benchmark
from climatenet.training.experiment_registry import ExperimentRegistry
from climatenet.utils.paths import resolve_project_path

ALLOWED_DRY_RUN_SPLITS = {
    "random_split",
    "spatial_block_holdout",
    "temporal_holdout",
}
MAX_DRY_RUN_MONTHS = 36
MAX_DRY_RUN_GRID_CELLS = 1_000
MAX_DRY_RUN_INPUT_BYTES = 512 * 1024 * 1024


def validate_era5_dry_run_config(config: dict[str, Any]) -> None:
    """Reject configs that could accidentally become a full experiment."""
    if config.get("dry_run") is not True:
        raise ValueError("ERA5 dry-run config must set dry_run: true")
    if bool(config.get("synthetic", False)):
        raise ValueError("ERA5 dry-run must set synthetic: false")
    if "era5" not in str(config.get("data_source", "")).casefold():
        raise ValueError("ERA5 dry-run data_source must explicitly identify ERA5")

    real_data = config.get("real_data")
    if not isinstance(real_data, dict):
        raise ValueError("ERA5 dry-run config requires a real_data mapping")
    paths = real_data.get("netcdf_paths", [])
    if not isinstance(paths, list) or not paths:
        raise ValueError("real_data.netcdf_paths must be a non-empty list")
    if not real_data.get("bbox"):
        raise ValueError(
            "ERA5 dry-run requires a bounding box; full-region reads are blocked"
        )
    max_grid_cells = int(real_data.get("max_grid_cells", 0))
    if not 0 < max_grid_cells <= MAX_DRY_RUN_GRID_CELLS:
        raise ValueError(
            f"real_data.max_grid_cells must be in 1..{MAX_DRY_RUN_GRID_CELLS}"
        )
    max_input_bytes = int(real_data.get("max_input_bytes", 0))
    if not 0 < max_input_bytes <= MAX_DRY_RUN_INPUT_BYTES:
        raise ValueError(
            "real_data.max_input_bytes must be positive and no larger than "
            f"{MAX_DRY_RUN_INPUT_BYTES:,}"
        )

    start = real_data.get("start")
    end = real_data.get("end")
    if not start or not end:
        raise ValueError("ERA5 dry-run requires real_data.start and real_data.end")
    start_period = pd.Period(str(start), freq="M")
    end_period = pd.Period(str(end), freq="M")
    month_count = end_period.ordinal - start_period.ordinal + 1
    if month_count <= 0 or month_count > MAX_DRY_RUN_MONTHS:
        raise ValueError(
            f"ERA5 dry-run window must contain 1..{MAX_DRY_RUN_MONTHS} "
            f"months, got {month_count}"
        )

    split_protocols = config.get("split_protocols", [])
    if (
        not isinstance(split_protocols, list)
        or not split_protocols
        or len(split_protocols) > 2
        or not set(split_protocols).issubset(ALLOWED_DRY_RUN_SPLITS)
    ):
        raise ValueError(
            "ERA5 dry-run allows at most two protocols chosen from "
            f"{sorted(ALLOWED_DRY_RUN_SPLITS)}"
        )
    models = config.get("models", [])
    if len(models) > 3:
        raise ValueError("ERA5 dry-run allows at most three traditional models")
    model_names = [
        model if isinstance(model, str) else model.get("name", "")
        for model in models
    ]
    if "tcn" in model_names:
        raise ValueError("TCN is prohibited in the ERA5 dry-run")
    for model in models:
        if not isinstance(model, dict):
            continue
        n_estimators = model.get("params", {}).get("n_estimators")
        if n_estimators is not None and int(n_estimators) > 100:
            raise ValueError(
                "ERA5 dry-run caps ensemble n_estimators at 100"
            )


def era5_audit_options(config: dict[str, Any]) -> dict[str, Any]:
    real_data = config["real_data"]
    return {
        "paths": real_data["netcdf_paths"],
        "start": str(real_data["start"]),
        "end": str(real_data["end"]),
        "bbox": real_data["bbox"],
        "max_grid_cells": int(real_data["max_grid_cells"]),
        "max_total_bytes": int(real_data["max_input_bytes"]),
        "input_window": int(config.get("input_window", 6)),
        "region": real_data.get("region"),
    }


def prepare_era5_dry_run_features(
    config: dict[str, Any],
    destination: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit, convert units and add row-wise features for a guarded subset."""
    validate_era5_dry_run_config(config)
    options = era5_audit_options(config)
    audit = audit_era5_files(**options)
    if audit.get("blocking_issues"):
        raise ValueError(
            "ERA5 readiness audit has blocking issues: "
            f"{audit['blocking_issues']}"
        )
    non_finite = sum(
        item["non_finite_count"]
        for item in audit["converted_variable_summary"].values()
    )
    if non_finite:
        raise ValueError(
            f"ERA5 audit found {non_finite:,} converted non-finite values; "
            "dry-run preparation refuses to silently drop or impute them"
        )

    frames = [
        preprocess_era5_file(
            resolve_project_path(path),
            start=options["start"],
            end=options["end"],
            bbox=options["bbox"],
            max_grid_cells=options["max_grid_cells"],
            region=options["region"],
            drop_invalid=False,
        )
        for path in options["paths"]
    ]
    raw_records = pd.concat(frames, ignore_index=True)
    features = add_physical_features(raw_records)
    numeric = features.select_dtypes(include=[np.number])
    non_finite_columns = {
        column: int((~np.isfinite(numeric[column])).sum())
        for column in numeric
        if (~np.isfinite(numeric[column])).any()
    }
    if non_finite_columns:
        raise ValueError(
            "Physical feature generation produced non-finite values: "
            f"{non_finite_columns}"
        )
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features, audit


def run_era5_dry_run(
    config: dict[str, Any],
    output_root: str | Path = "outputs/benchmark_runs",
) -> ExperimentRegistry:
    """Prepare a bounded real subset and execute the formal benchmark runner."""
    validate_era5_dry_run_config(config)
    staging_dir = Path(
        tempfile.mkdtemp(prefix="climatenet-era5-dry-run-")
    )
    features_path = staging_dir / "era5_dry_run_features.csv"
    _, audit = prepare_era5_dry_run_features(config, features_path)

    resolved_config = copy.deepcopy(config)
    resolved_config["features_path"] = str(features_path)
    resolved_config["real_data"]["audit_status"] = audit["status"]
    resolved_config["real_data"]["prepared_row_count"] = audit[
        "converted_row_count"
    ]
    registry = run_benchmark(resolved_config, output_root=output_root)
    run_dir = registry.path.parent
    audit_path = run_dir / "data_audit" / "era5_readiness.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_features_path = (
        audit_path.parent / "era5_dry_run_physical_features.csv"
    )
    shutil.copy2(features_path, prepared_features_path)
    with audit_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)

    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["real_data_audit"] = {
        "path": str(audit_path.relative_to(run_dir)),
        "status": audit["status"],
        "prepared_features_artifact": str(
            prepared_features_path.relative_to(run_dir)
        ),
        "prepared_features_sha256": metadata["data_files"].get(
            str(features_path.resolve())
        ),
        "source_files": [
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in audit["input_files"]
        ],
        "converted_row_count": audit["converted_row_count"],
        "lag_sample_estimate": audit["lag_sample_estimate"],
        "warnings": audit["warnings"],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return registry
