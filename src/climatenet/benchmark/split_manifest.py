"""Fixed split-manifest generation for reproducible benchmarks.

Each split function reads a forecasting-sample DataFrame and writes
``train.csv``, ``val.csv``, and ``test.csv`` into the provided output
directory.  The CSVs share a common schema::

    sample_id, year, month, lat, lon, region, climate_zone, split

where ``split`` is one of ``"train"``, ``"val"``, or ``"test"``.

Split types
-----------
1. **random** — sample-level shuffle, noisy baseline.
2. **temporal_holdout** — train on earlier years, test on future years.
3. **climate_zone_holdout** — train on some climate zones, test on others.
4. **spatial_temporal_holdout** — combined spatial-block + temporal holdout.

Every split function also returns a metadata dict and the audit result
from :func:`audit_split_manifest`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# Columns written to every manifest CSV.
_MANIFEST_COLS = ["sample_id", "year", "month", "lat", "lon", "region", "climate_zone", "split"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has the canonical column names.

    Accepts ``latitude``/``longitude`` as aliases for ``lat``/``lon``.
    """
    out = df.copy()
    if "lat" not in out.columns and "latitude" in out.columns:
        out["lat"] = out["latitude"]
    if "lon" not in out.columns and "longitude" in out.columns:
        out["lon"] = out["longitude"]
    return out


def _check_required_columns(df: pd.DataFrame, *, label: str = "DataFrame") -> None:
    """Raise ``ValueError`` if mandatory columns are missing."""
    required = {"sample_id", "year", "month", "lat", "lon", "region", "climate_zone"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


def _build_manifest(
    df: pd.DataFrame,
    ids: list[str],
    split_label: str,
) -> pd.DataFrame:
    """Return a manifest DataFrame for the given sample IDs."""
    subset = df[df["sample_id"].isin(ids)].copy()
    subset["split"] = split_label
    return subset[_MANIFEST_COLS]


def _save_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write train.csv, val.csv, test.csv to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


# ---------------------------------------------------------------------------
# 1. Random split
# ---------------------------------------------------------------------------


def generate_random_split(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Random sample-level split — optimistic baseline only.

    .. warning::
       This split leaks spatial and temporal information and must never
       be reported as the primary benchmark result.
    """
    df = _normalise_columns(df)
    _check_required_columns(df)

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_frac + val_frac + test_frac must equal 1.0, got {total}"
        )

    output_dir = Path(output_dir)
    all_ids = df["sample_id"].tolist()
    n = len(all_ids)
    if n < 3:
        raise ValueError(f"Need at least 3 samples for a split, got {n}.")

    # train vs (val + test)
    val_test_frac = val_frac + test_frac
    train_ids, remaining = train_test_split(
        all_ids, test_size=val_test_frac, random_state=seed, shuffle=True,
    )

    # val vs test from remainder
    rel_val_frac = val_frac / val_test_frac if val_test_frac > 0 else 0.5
    val_ids, test_ids = train_test_split(
        remaining, test_size=1.0 - rel_val_frac, random_state=seed, shuffle=True,
    )

    train_manifest = _build_manifest(df, train_ids, "train")
    val_manifest = _build_manifest(df, val_ids, "val")
    test_manifest = _build_manifest(df, test_ids, "test")

    _save_manifests(train_manifest, val_manifest, test_manifest, output_dir)

    metadata = {
        "split_type": "random",
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
    }
    _save_metadata(metadata, output_dir)

    audit = audit_split_manifest(output_dir, "random")

    return {"metadata": metadata, "audit": audit}


# ---------------------------------------------------------------------------
# 2. Temporal holdout
# ---------------------------------------------------------------------------


def generate_temporal_holdout_split(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    train_years: list[int] | None = None,
    val_years: list[int] | None = None,
    test_years: list[int] | None = None,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Temporal holdout — train on earlier years, test on future years.

    Years specified in ``test_years`` must not appear in ``train_years``
    or ``val_years``.
    """
    df = _normalise_columns(df)
    _check_required_columns(df)

    output_dir = Path(output_dir)

    years = sorted(df["year"].unique())
    n_years = len(years)

    if n_years < 3:
        raise ValueError(f"Need at least 3 unique years, got {n_years}: {years}")

    if train_years is None and val_years is None and test_years is None:
        # Auto-assign: earliest 60% train, next 20% val, latest 20% test
        n_train = max(1, int(n_years * 0.6))
        n_val = max(1, int(n_years * 0.2))
        n_test = n_years - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_val = n_years - n_train - n_test
            if n_val < 1:
                n_val = 1
                n_train = n_years - n_val - n_test

        train_years = years[:n_train]
        val_years = years[n_train : n_train + n_val]
        test_years = years[n_train + n_val : n_train + n_val + n_test]

    # Validate no overlap
    train_set = set(train_years or [])
    val_set = set(val_years or [])
    test_set = set(test_years or [])

    if train_set & test_set:
        raise ValueError(
            f"Temporal leakage: test years {sorted(test_set & train_set)} "
            f"appear in train_years."
        )
    if val_set & test_set:
        raise ValueError(
            f"Temporal leakage: test years {sorted(test_set & val_set)} "
            f"appear in val_years."
        )

    train_ids = df[df["year"].isin(train_set)]["sample_id"].tolist()
    val_ids = df[df["year"].isin(val_set)]["sample_id"].tolist()
    test_ids = df[df["year"].isin(test_set)]["sample_id"].tolist()

    if not train_ids:
        raise ValueError(f"No samples found for train_years={sorted(train_set)}")
    if not val_ids:
        raise ValueError(f"No samples found for val_years={sorted(val_set)}")
    if not test_ids:
        raise ValueError(f"No samples found for test_years={sorted(test_set)}")

    train_manifest = _build_manifest(df, train_ids, "train")
    val_manifest = _build_manifest(df, val_ids, "val")
    test_manifest = _build_manifest(df, test_ids, "test")

    _save_manifests(train_manifest, val_manifest, test_manifest, output_dir)

    metadata = {
        "split_type": "temporal_holdout",
        "seed": seed,
        "train_years": sorted(train_set),
        "val_years": sorted(val_set),
        "test_years": sorted(test_set),
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
    }
    _save_metadata(metadata, output_dir)

    audit = audit_split_manifest(output_dir, "temporal_holdout")

    return {"metadata": metadata, "audit": audit}


# ---------------------------------------------------------------------------
# 3. Climate-zone holdout
# ---------------------------------------------------------------------------


def generate_climate_zone_holdout_split(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    train_zones: list[str] | None = None,
    val_zones: list[str] | None = None,
    test_zones: list[str] | None = None,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Climate-zone holdout — train on some climate zones, test on others.

    Climate zones in ``test_zones`` must not appear in ``train_zones``
    or ``val_zones``.
    """
    df = _normalise_columns(df)
    _check_required_columns(df)

    output_dir = Path(output_dir)

    zones = sorted(df["climate_zone"].unique())
    n_zones = len(zones)

    if n_zones < 2:
        raise ValueError(
            f"Need at least 2 unique climate zones, got {n_zones}: {zones}"
        )

    if train_zones is None and val_zones is None and test_zones is None:
        # Auto-assign: train on all but 1 zone, test on remaining zone
        # Reserve one zone for val if possible
        if n_zones >= 3:
            test_zones = [zones[-1]]
            val_zones = [zones[-2]]
            train_zones = zones[:-2]
        else:
            test_zones = [zones[-1]]
            val_zones = [zones[0]]
            train_zones = [zones[0]]

    train_set = set(train_zones or [])
    val_set = set(val_zones or [])
    test_set = set(test_zones or [])

    if train_set & test_set:
        raise ValueError(
            f"Climate-zone leakage: test zones {sorted(test_set & train_set)} "
            f"appear in train_zones."
        )

    train_ids = df[df["climate_zone"].isin(train_set)]["sample_id"].tolist()
    val_ids = df[df["climate_zone"].isin(val_set)]["sample_id"].tolist()
    test_ids = df[df["climate_zone"].isin(test_set)]["sample_id"].tolist()

    if not train_ids:
        raise ValueError(f"No samples found for train_zones={sorted(train_set)}")
    if not val_ids:
        raise ValueError(f"No samples found for val_zones={sorted(val_set)}")
    if not test_ids:
        raise ValueError(f"No samples found for test_zones={sorted(test_set)}")

    train_manifest = _build_manifest(df, train_ids, "train")
    val_manifest = _build_manifest(df, val_ids, "val")
    test_manifest = _build_manifest(df, test_ids, "test")

    _save_manifests(train_manifest, val_manifest, test_manifest, output_dir)

    metadata = {
        "split_type": "climate_zone_holdout",
        "seed": seed,
        "train_zones": sorted(train_set),
        "val_zones": sorted(val_set),
        "test_zones": sorted(test_set),
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
    }
    _save_metadata(metadata, output_dir)

    audit = audit_split_manifest(output_dir, "climate_zone_holdout")

    return {"metadata": metadata, "audit": audit}


# ---------------------------------------------------------------------------
# 4. Spatial-temporal holdout
# ---------------------------------------------------------------------------


def generate_spatial_temporal_holdout_split(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    train_regions: list[str] | None = None,
    val_regions: list[str] | None = None,
    test_regions: list[str] | None = None,
    train_years: list[int] | None = None,
    val_years: list[int] | None = None,
    test_years: list[int] | None = None,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Spatial-temporal holdout — joint spatial + temporal constraint.

    Test regions and test years must not appear in train or val sets.
    This is the strictest split protocol.
    """
    df = _normalise_columns(df)
    _check_required_columns(df)

    output_dir = Path(output_dir)

    regions = sorted(df["region"].unique())
    years = sorted(df["year"].unique())

    if len(regions) < 2:
        raise ValueError(f"Need at least 2 regions, got {len(regions)}: {regions}")
    if len(years) < 3:
        raise ValueError(f"Need at least 3 years, got {len(years)}: {years}")

    # Auto-assign
    if train_regions is None and test_regions is None:
        if len(regions) >= 3:
            test_regions = [regions[-1]]
            val_regions = [regions[-2]]
            train_regions = regions[:-2]
        else:
            test_regions = [regions[-1]]
            val_regions = [regions[0]]
            train_regions = [regions[0]]

    if train_years is None and test_years is None:
        n_years = len(years)
        n_train_y = max(1, int(n_years * 0.6))
        n_val_y = max(1, int(n_years * 0.2))
        n_test_y = n_years - n_train_y - n_val_y
        if n_test_y < 1:
            n_test_y = 1
            n_val_y = n_years - n_train_y - n_test_y
            if n_val_y < 1:
                n_val_y = 1
                n_train_y = n_years - n_val_y - n_test_y

        train_years = years[:n_train_y]
        val_years = years[n_train_y : n_train_y + n_val_y]
        test_years = years[n_train_y + n_val_y : n_train_y + n_val_y + n_test_y]

    train_r_set = set(train_regions or [])
    val_r_set = set(val_regions or [])
    test_r_set = set(test_regions or [])
    train_y_set = set(train_years or [])
    val_y_set = set(val_years or [])
    test_y_set = set(test_years or [])

    # Validate spatial constraint
    if train_r_set & test_r_set:
        raise ValueError(
            f"Spatial leakage: test regions {sorted(test_r_set & train_r_set)} "
            f"appear in train_regions."
        )

    # Validate temporal constraint
    if train_y_set & test_y_set:
        raise ValueError(
            f"Temporal leakage: test years {sorted(test_y_set & train_y_set)} "
            f"appear in train_years."
        )

    train_ids = df[
        df["region"].isin(train_r_set) & df["year"].isin(train_y_set)
    ]["sample_id"].tolist()
    val_ids = df[
        df["region"].isin(val_r_set) & df["year"].isin(val_y_set)
    ]["sample_id"].tolist()
    test_ids = df[
        df["region"].isin(test_r_set) & df["year"].isin(test_y_set)
    ]["sample_id"].tolist()

    if not train_ids:
        raise ValueError(
            f"No samples for train_regions={sorted(train_r_set)} × "
            f"train_years={sorted(train_y_set)}"
        )
    if not val_ids:
        raise ValueError(
            f"No samples for val_regions={sorted(val_r_set)} × "
            f"val_years={sorted(val_y_set)}"
        )
    if not test_ids:
        raise ValueError(
            f"No samples for test_regions={sorted(test_r_set)} × "
            f"test_years={sorted(test_y_set)}"
        )

    train_manifest = _build_manifest(df, train_ids, "train")
    val_manifest = _build_manifest(df, val_ids, "val")
    test_manifest = _build_manifest(df, test_ids, "test")

    _save_manifests(train_manifest, val_manifest, test_manifest, output_dir)

    metadata = {
        "split_type": "spatial_temporal_holdout",
        "seed": seed,
        "train_regions": sorted(train_r_set),
        "val_regions": sorted(val_r_set),
        "test_regions": sorted(test_r_set),
        "train_years": sorted(train_y_set),
        "val_years": sorted(val_y_set),
        "test_years": sorted(test_y_set),
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
    }
    _save_metadata(metadata, output_dir)

    audit = audit_split_manifest(output_dir, "spatial_temporal_holdout")

    return {"metadata": metadata, "audit": audit}


# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------


def _save_metadata(metadata: dict[str, Any], output_dir: Path) -> None:
    """Write ``split_metadata.json`` alongside the manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_split_manifest(
    split_dir: str | Path,
    split_type: str,
) -> dict[str, Any]:
    """Audit a split manifest directory for leakage and validity.

    Reads ``train.csv``, ``val.csv``, and ``test.csv`` from ``split_dir``
    and checks:

    - No sample_id overlap across train/val/test.
    - All three sets are non-empty.
    - Temporal: test years do not appear in train (for temporal splits).
    - Climate-zone: test zones do not appear in train (for climate-zone splits).
    - Spatial: test regions do not appear in train (for spatial-temporal splits).

    Parameters
    ----------
    split_dir
        Directory containing ``train.csv``, ``val.csv``, ``test.csv``.
    split_type
        One of ``"random"``, ``"temporal_holdout"``,
        ``"climate_zone_holdout"``, ``"spatial_temporal_holdout"``.

    Returns
    -------
    dict
        Audit result with keys ``sample_overlap``, ``temporal_overlap``,
        ``climate_zone_overlap``, ``spatial_overlap``, ``train_size``,
        ``val_size``, ``test_size``, and ``warnings``.
    """
    split_dir = Path(split_dir)
    warnings: list[str] = []

    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"

    for label, p in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not p.exists():
            return {
                "sample_overlap": "FAIL",
                "temporal_overlap": "FAIL",
                "climate_zone_overlap": "FAIL",
                "spatial_overlap": "FAIL",
                "train_size": 0,
                "val_size": 0,
                "test_size": 0,
                "warnings": [f"Missing file: {p.name}"],
            }

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_ids = set(train_df["sample_id"])
    val_ids = set(val_df["sample_id"])
    test_ids = set(test_df["sample_id"])

    train_size = len(train_ids)
    val_size = len(val_ids)
    test_size = len(test_ids)

    # --- sample overlap ----------------------------------------------------
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    sample_overlap = "PASS" if not (overlap_tv or overlap_tt or overlap_vt) else "FAIL"
    if sample_overlap == "FAIL":
        details = []
        if overlap_tv:
            details.append(f"train∩val={len(overlap_tv)}")
        if overlap_tt:
            details.append(f"train∩test={len(overlap_tt)}")
        if overlap_vt:
            details.append(f"val∩test={len(overlap_vt)}")
        warnings.append(f"Sample overlap detected: {', '.join(details)}")

    # --- empty checks ------------------------------------------------------
    if train_size == 0:
        warnings.append("train set is empty")
    if val_size == 0:
        warnings.append("val set is empty")
    if test_size == 0:
        warnings.append("test set is empty")

    # --- temporal overlap --------------------------------------------------
    temporal_overlap = "N/A"
    if split_type in ("temporal_holdout", "spatial_temporal_holdout"):
        train_years = set(train_df["year"].unique())
        test_years = set(test_df["year"].unique())
        if train_years & test_years:
            temporal_overlap = "FAIL"
            warnings.append(
                f"Temporal leakage: test years {sorted(train_years & test_years)} "
                f"appear in train."
            )
        else:
            temporal_overlap = "PASS"

    # --- climate-zone overlap ----------------------------------------------
    climate_zone_overlap = "N/A"
    if split_type == "climate_zone_holdout":
        train_zones = set(train_df["climate_zone"].unique())
        test_zones = set(test_df["climate_zone"].unique())
        if train_zones & test_zones:
            climate_zone_overlap = "FAIL"
            warnings.append(
                f"Climate-zone leakage: test zones "
                f"{sorted(train_zones & test_zones)} appear in train."
            )
        else:
            climate_zone_overlap = "PASS"

    # --- spatial overlap ---------------------------------------------------
    spatial_overlap = "N/A"
    if split_type == "spatial_temporal_holdout":
        train_regions = set(train_df["region"].unique())
        test_regions = set(test_df["region"].unique())
        if train_regions & test_regions:
            spatial_overlap = "FAIL"
            warnings.append(
                f"Spatial leakage: test regions "
                f"{sorted(train_regions & test_regions)} appear in train."
            )
        else:
            spatial_overlap = "PASS"

    return {
        "sample_overlap": sample_overlap,
        "temporal_overlap": temporal_overlap,
        "climate_zone_overlap": climate_zone_overlap,
        "spatial_overlap": spatial_overlap,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "warnings": warnings,
    }
