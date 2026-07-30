"""Adapter from audited repeated-spatial block manifests to SplitResult."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.benchmark.split_protocols import SplitResult, save_split_result


PROTOCOL = "repeated_region_stratified_spatial"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_ids(
    latitude: pd.Series,
    longitude: pd.Series,
    block_size_deg: float,
) -> pd.Series:
    lat = (
        np.floor(latitude.to_numpy(dtype=np.float64) / block_size_deg)
        * block_size_deg
    ).astype(int)
    lon = (
        np.floor(longitude.to_numpy(dtype=np.float64) / block_size_deg)
        * block_size_deg
    ).astype(int)
    return pd.Series(
        [f"block_lat{x}_lon{y}" for x, y in zip(lat, lon)],
        index=latitude.index,
        dtype="string",
    )


def load_manifest_backed_repeated_splits(
    samples_df: pd.DataFrame,
    repeated_config: dict[str, Any],
    output_root: str | Path,
) -> list[SplitResult]:
    """Load accepted folds and expand block roles to forecasting sample IDs."""
    required = {
        "sample_id",
        "grid_id",
        "region",
        "latitude",
        "longitude",
    }
    missing = required - set(samples_df.columns)
    if missing:
        raise ValueError(
            f"Repeated spatial adapter missing sample columns: {sorted(missing)}"
        )
    audit_path = Path(repeated_config["audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_status = repeated_config.get("required_audit_status", "ready")
    if audit.get("status") != required_status or not audit.get(
        "audit_passed", False
    ):
        raise ValueError(
            "Repeated spatial audit is not accepted: "
            f"status={audit.get('status')!r}, "
            f"audit_passed={audit.get('audit_passed')!r}"
        )
    audit_folds = {
        int(item["fold"]): item for item in audit.get("folds", [])
    }
    folds = [int(fold) for fold in repeated_config["folds"]]
    if len(folds) != len(set(folds)):
        raise ValueError("Repeated spatial folds contain duplicates")
    if any(
        fold not in audit_folds or not audit_folds[fold].get("fold_passed")
        for fold in folds
    ):
        raise ValueError("Every requested fold must pass the composition audit")

    manifest_dir = Path(repeated_config["manifest_dir"])
    block_size = float(repeated_config["block_size_deg"])
    generation_seed = int(repeated_config["generation_seed"])
    grids = samples_df[
        ["region", "grid_id", "latitude", "longitude"]
    ].drop_duplicates(["region", "grid_id"])
    grids = grids.copy()
    grids["spatial_block_id"] = _block_ids(
        grids["latitude"], grids["longitude"], block_size
    )
    sample_regions = samples_df["region"].astype("string")
    results: list[SplitResult] = []
    audit_hash = _sha256(audit_path)

    for fold in folds:
        assignment_path = (
            manifest_dir / f"fold_{fold}_block_assignments.csv"
        )
        assignment = pd.read_csv(assignment_path)
        required_assignment = {
            "region",
            "spatial_block_id",
            "partition",
        }
        if not required_assignment.issubset(assignment):
            raise ValueError(
                f"Fold {fold} manifest missing columns: "
                f"{sorted(required_assignment-set(assignment.columns))}"
            )
        if assignment.duplicated(["region", "spatial_block_id"]).any():
            raise ValueError(f"Fold {fold} has duplicate block assignments")
        if set(assignment["partition"]) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"Fold {fold} has invalid partition labels")

        grid_roles = grids.merge(
            assignment[
                ["region", "spatial_block_id", "partition"]
            ],
            on=["region", "spatial_block_id"],
            how="left",
            validate="many_to_one",
        )
        if grid_roles["partition"].isna().any():
            raise ValueError(f"Fold {fold} leaves grid cells unassigned")
        if grid_roles.duplicated(["region", "grid_id"]).any():
            raise ValueError(f"Fold {fold} assigns a grid more than once")

        row_partition = pd.Series(
            index=samples_df.index, dtype="string"
        )
        for region, regional in grid_roles.groupby(
            "region", observed=True
        ):
            mapping = regional.set_index("grid_id")["partition"].to_dict()
            mask = sample_regions.eq(str(region))
            row_partition.loc[mask] = (
                samples_df.loc[mask, "grid_id"].astype("string").map(mapping)
            )
        if row_partition.isna().any():
            raise ValueError(f"Fold {fold} leaves samples unassigned")

        partition_by_grid = (
            pd.DataFrame(
                {
                    "region": sample_regions,
                    "grid_id": samples_df["grid_id"].astype("string"),
                    "partition": row_partition,
                }
            )
            .drop_duplicates()
            .groupby(["region", "grid_id"], observed=True)["partition"]
            .nunique()
        )
        if int((partition_by_grid > 1).sum()) != 0:
            raise ValueError(f"Fold {fold} has grid-cell partition leakage")

        ids = {
            partition: samples_df.loc[
                row_partition.eq(partition), "sample_id"
            ].tolist()
            for partition in ["train", "validation", "test"]
        }
        split_id = f"{PROTOCOL}_fold_{fold}"
        result = SplitResult(
            split_id=split_id,
            protocol=PROTOCOL,
            train_ids=ids["train"],
            val_ids=ids["validation"],
            test_ids=ids["test"],
            config={
                "seed": generation_seed,
                "fold": fold,
                "block_size_deg": block_size,
                "manifest_path": str(assignment_path.resolve()),
            },
            metadata={
                "fold": fold,
                "fold_audit_path": str(audit_path.resolve()),
                "fold_audit_sha256": audit_hash,
                "fold_audit_status": audit["status"],
                "fold_acceptance": audit_folds[fold],
                "manifest_path": str(assignment_path.resolve()),
                "manifest_sha256": _sha256(assignment_path),
                "generation_seed": generation_seed,
                "block_size_deg": block_size,
                "assignment_uses_target": False,
                "grid_leakage_count": 0,
                "n_train": len(ids["train"]),
                "n_validation": len(ids["validation"]),
                "n_test": len(ids["test"]),
            },
        )
        save_split_result(result, Path(output_root) / split_id)
        results.append(result)
    return results
