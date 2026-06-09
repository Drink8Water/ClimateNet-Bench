"""Benchmark split protocols for spatio-temporal climate forecasting.

Implements six split protocols of increasing difficulty:

1. **random** — optimistic baseline (sample-level shuffle)
2. **spatial_block** — hold out entire spatial blocks
3. **temporal** — train on earlier years, test on future years
4. **region_transfer** — train on some regions, test on disjoint regions
5. **climate_zone_transfer** — train on some climate zones, test on held-out zone
6. **spatiotemporal** — joint spatial-block + temporal holdout

Every split returns **sample IDs** (not data) so that the same underlying
DataFrame can be shared across splits without duplication.

Anti-leakage guarantees
-----------------------
- Random split is the ONLY protocol that allows any overlap. It is
  labelled ``"random"`` in metadata and should never be reported as the
  primary result.
- All other protocols enforce disjointness at the appropriate level
  (grid_id, year, region, climate_type, or a combination).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

ALL_SPLIT_NAMES: list[str] = [
    "random",
    "spatial_block",
    "temporal",
    "region_transfer",
    "climate_zone_transfer",
    "spatiotemporal",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """Result of one split protocol execution.

    Attributes
    ----------
    split_id
        Unique split identifier (e.g. ``"spatial_block_sahara_5deg"``).
    protocol
        Split protocol name (e.g. ``"spatial_block"``).
    train_ids
        List of ``sample_id`` values for training.
    val_ids
        List of ``sample_id`` values for validation.
    test_ids
        List of ``sample_id`` values for testing.
    metadata
        Arbitrary metadata dict (serialisable to JSON).
    config
        The config dict used to produce this split.
    """

    split_id: str
    protocol: str
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def all_ids(self) -> list[str]:
        return self.train_ids + self.val_ids + self.test_ids


# ---------------------------------------------------------------------------
# Spatial block helper
# ---------------------------------------------------------------------------


def _make_spatial_block_id(
    lat: float, lon: float, block_size_deg: float = 5.0
) -> str:
    """Assign a spatial block ID from rounded lat/lon.

    >>> _make_spatial_block_id(23.4, 115.8, 5.0)
    'block_lat20_lon115'
    """
    lat_block = int(np.floor(lat / block_size_deg) * block_size_deg)
    lon_block = int(np.floor(lon / block_size_deg) * block_size_deg)
    return f"block_lat{lat_block}_lon{lon_block}"


# ---------------------------------------------------------------------------
# Protocol 1 — Random split (optimistic baseline)
# ---------------------------------------------------------------------------


def make_random_split(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = RANDOM_SEED,
    split_id: str = "random",
) -> SplitResult:
    """Random sample-level split — optimistic baseline ONLY.

    .. warning::
       This split leaks spatial and temporal information.  It must be
       clearly labelled as a baseline and must NOT be reported as the
       primary benchmark result.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    all_ids = df["sample_id"].tolist()
    n = len(all_ids)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_ids, remaining = train_test_split(
        all_ids,
        test_size=val_test_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Second split: val vs test from the remainder
    val_rel_ratio = val_ratio / val_test_ratio if val_test_ratio > 0 else 0.5
    val_ids, test_ids = train_test_split(
        remaining,
        test_size=1.0 - val_rel_ratio,
        random_state=seed,
        shuffle=True,
    )

    return SplitResult(
        split_id=split_id,
        protocol="random",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        metadata={
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "note": "OPTIMISTIC BASELINE — leaks spatial and temporal information. "
            "Do NOT report as primary result.",
        },
        config={
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Protocol 2 — Spatial block holdout
# ---------------------------------------------------------------------------


def make_spatial_block_split(
    df: pd.DataFrame,
    block_size_deg: float = 5.0,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = RANDOM_SEED,
    split_id: str = "spatial_block",
) -> SplitResult:
    """Split by spatial blocks — no block overlap between train and test.

    Each sample is assigned to a spatial block based on its (latitude,
    longitude).  Blocks are then randomly split into train / val / test.
    """
    df = df.copy()
    if "spatial_block_id" not in df.columns:
        df["spatial_block_id"] = df.apply(
            lambda r: _make_spatial_block_id(r["latitude"], r["longitude"], block_size_deg),
            axis=1,
        )

    blocks = df[["spatial_block_id"]].drop_duplicates()
    n_blocks = len(blocks)

    if n_blocks < 3:
        raise ValueError(
            f"Need at least 3 spatial blocks for train/val/test split, got {n_blocks}. "
            f"Try a smaller block_size_deg (current: {block_size_deg})."
        )

    # Split blocks into train / val / test.
    # First: train vs (val + test).
    val_test_ratio = val_ratio + (1.0 - train_ratio - val_ratio)
    # Actually simpler: val_test_ratio = 1.0 - train_ratio
    val_test_ratio = 1.0 - train_ratio
    train_blocks, remaining_blocks = train_test_split(
        blocks,
        test_size=val_test_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Second: split remaining into val and test.
    # With small block counts, float ratios can produce empty splits.
    # Use integer counts when possible.
    n_remaining = len(remaining_blocks)
    if val_ratio > 0 and n_remaining >= 2:
        test_denom = 1.0 - train_ratio
        val_rel = val_ratio / test_denom if test_denom > 0 else 0.5

        # Clamp to avoid float-edge empty splits
        n_val = max(1, min(n_remaining - 1, int(np.round(val_rel * n_remaining))))
        n_test = n_remaining - n_val

        remaining_idx = remaining_blocks.index.tolist()
        rng = np.random.default_rng(seed)
        rng.shuffle(remaining_idx)
        val_idx = remaining_idx[:n_val]
        test_idx = remaining_idx[n_val:]

        val_blocks = remaining_blocks.loc[val_idx]
        test_blocks = remaining_blocks.loc[test_idx]
    else:
        val_blocks = pd.DataFrame(columns=blocks.columns)
        test_blocks = remaining_blocks

    train_ids = df[df["spatial_block_id"].isin(train_blocks["spatial_block_id"])]["sample_id"].tolist()
    val_ids = df[df["spatial_block_id"].isin(val_blocks["spatial_block_id"])]["sample_id"].tolist()
    test_ids = df[df["spatial_block_id"].isin(test_blocks["spatial_block_id"])]["sample_id"].tolist()

    return SplitResult(
        split_id=split_id,
        protocol="spatial_block",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        metadata={
            "block_size_deg": block_size_deg,
            "n_blocks_total": n_blocks,
            "n_blocks_train": len(train_blocks),
            "n_blocks_val": len(val_blocks),
            "n_blocks_test": len(test_blocks),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        },
        config={
            "block_size_deg": block_size_deg,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Protocol 3 — Temporal holdout
# ---------------------------------------------------------------------------


def make_temporal_split(
    df: pd.DataFrame,
    train_years: list[int] | None = None,
    val_year: int | None = None,
    test_year: int | None = None,
    seed: int = RANDOM_SEED,
    split_id: str = "temporal",
) -> SplitResult:
    """Temporal holdout — train on earlier years, test on future year(s).

    No ``target_year`` in val or test may appear in train.

    Default: train=[2019,2020,2021], val=2022, test=2023.
    """
    if train_years is None:
        train_years = [2019, 2020, 2021]
    if val_year is None:
        val_year = 2022
    if test_year is None:
        test_year = 2023

    train_ids = df[df["target_year"].isin(train_years)]["sample_id"].tolist()
    val_ids = df[df["target_year"] == val_year]["sample_id"].tolist()
    test_ids = df[df["target_year"] == test_year]["sample_id"].tolist()

    if not train_ids:
        raise ValueError(f"No samples found for train_years={train_years}")
    if not test_ids:
        raise ValueError(f"No samples found for test_year={test_year}")

    return SplitResult(
        split_id=split_id,
        protocol="temporal",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        metadata={
            "train_years": train_years,
            "val_year": val_year,
            "test_year": test_year,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        },
        config={
            "train_years": train_years,
            "val_year": val_year,
            "test_year": test_year,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Protocol 4 — Region transfer
# ---------------------------------------------------------------------------


def make_region_transfer_split(
    df: pd.DataFrame,
    train_regions: list[str],
    test_regions: list[str],
    val_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
    split_id: str = "region_transfer",
) -> SplitResult:
    """Region transfer — train on some regions, test on disjoint regions.

    Train and test region sets must have **zero overlap**.
    """
    train_set = set(train_regions)
    test_set = set(test_regions)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(
            f"Region overlap between train and test: {sorted(overlap)}"
        )

    train_df = df[df["region"].isin(train_regions)]
    test_df = df[df["region"].isin(test_regions)]

    if train_df.empty:
        raise ValueError(f"No samples found for train_regions={train_regions}")
    if test_df.empty:
        raise ValueError(f"No samples found for test_regions={test_regions}")

    train_ids_all = train_df["sample_id"].tolist()
    test_ids = test_df["sample_id"].tolist()

    # Carve out a validation set from training regions
    if val_ratio > 0 and len(train_ids_all) > 2:
        train_ids, val_ids = train_test_split(
            train_ids_all,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_ids = train_ids_all
        val_ids = []

    return SplitResult(
        split_id=split_id,
        protocol="region_transfer",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        metadata={
            "train_regions": sorted(train_set),
            "test_regions": sorted(test_set),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        },
        config={
            "train_regions": train_regions,
            "test_regions": test_regions,
            "val_ratio": val_ratio,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Protocol 5 — Climate-zone transfer
# ---------------------------------------------------------------------------


def make_climate_zone_transfer_split(
    df: pd.DataFrame,
    train_zones: list[str],
    test_zones: list[str],
    val_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
    split_id: str = "climate_zone_transfer",
) -> SplitResult:
    """Climate-zone transfer — train on some zones, test on held-out zones.

    Train and test climate-zone sets must have **zero overlap**.
    """
    if "climate_type" not in df.columns:
        raise ValueError(
            "DataFrame must have a 'climate_type' column for climate-zone transfer. "
            "Build the forecasting dataset with a RegionRegistry to inject climate types."
        )

    train_set = set(train_zones)
    test_set = set(test_zones)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(
            f"Climate-zone overlap between train and test: {sorted(overlap)}"
        )

    train_df = df[df["climate_type"].isin(train_zones)]
    test_df = df[df["climate_type"].isin(test_zones)]

    if train_df.empty:
        raise ValueError(f"No samples found for train_zones={train_zones}")
    if test_df.empty:
        raise ValueError(f"No samples found for test_zones={test_zones}")

    train_ids_all = train_df["sample_id"].tolist()
    test_ids = test_df["sample_id"].tolist()

    if val_ratio > 0 and len(train_ids_all) > 2:
        train_ids, val_ids = train_test_split(
            train_ids_all,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_ids = train_ids_all
        val_ids = []

    return SplitResult(
        split_id=split_id,
        protocol="climate_zone_transfer",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        metadata={
            "train_zones": sorted(train_set),
            "test_zones": sorted(test_set),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        },
        config={
            "train_zones": train_zones,
            "test_zones": test_zones,
            "val_ratio": val_ratio,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Protocol 6 — Spatiotemporal holdout (strictest)
# ---------------------------------------------------------------------------


def make_spatiotemporal_split(
    df: pd.DataFrame,
    block_size_deg: float = 5.0,
    train_years: list[int] | None = None,
    test_year: int | None = None,
    val_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
    split_id: str = "spatiotemporal",
) -> SplitResult:
    """Joint spatial-block + temporal holdout.

    - Test set = held-out spatial blocks × held-out year(s).
    - Train set = remaining blocks × remaining years.
    - No grid_id overlap between train and test.
    - No target_year leakage.
    """
    if train_years is None:
        train_years = [2019, 2020, 2021]
    if test_year is None:
        test_year = 2023

    df = df.copy()
    if "spatial_block_id" not in df.columns:
        df["spatial_block_id"] = df.apply(
            lambda r: _make_spatial_block_id(r["latitude"], r["longitude"], block_size_deg),
            axis=1,
        )

    all_blocks = df["spatial_block_id"].unique().tolist()
    if len(all_blocks) < 2:
        raise ValueError(
            f"Need at least 2 spatial blocks for spatiotemporal split, got {len(all_blocks)}."
        )

    # Hold out ~test_ratio of blocks for testing
    test_block_ratio = 0.3
    train_blocks_arr, test_blocks_arr = train_test_split(
        all_blocks,
        test_size=test_block_ratio,
        random_state=seed,
        shuffle=True,
    )
    train_block_set = set(train_blocks_arr)
    test_block_set = set(test_blocks_arr)

    # Train: train blocks × train years
    train_mask = (
        df["spatial_block_id"].isin(train_block_set)
        & df["target_year"].isin(train_years)
    )
    train_df = df[train_mask]

    # Test: test blocks × test year
    test_mask = (
        df["spatial_block_id"].isin(test_block_set)
        & (df["target_year"] == test_year)
    )
    test_df = df[test_mask]

    if train_df.empty:
        raise ValueError(
            "Spatiotemporal split produced empty train set. "
            "Check block_size_deg and train_years."
        )
    if test_df.empty:
        raise ValueError(
            "Spatiotemporal split produced empty test set. "
            "Check block_size_deg and test_year."
        )

    train_ids_all = train_df["sample_id"].tolist()
    test_ids = test_df["sample_id"].tolist()

    if val_ratio > 0 and len(train_ids_all) > 2:
        train_ids, val_ids = train_test_split(
            train_ids_all,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_ids = train_ids_all
        val_ids = []

    return SplitResult(
        split_id=split_id,
        protocol="spatiotemporal",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        metadata={
            "block_size_deg": block_size_deg,
            "train_years": train_years,
            "test_year": test_year,
            "n_blocks_total": len(all_blocks),
            "n_blocks_train": len(train_block_set),
            "n_blocks_test": len(test_block_set),
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        },
        config={
            "block_size_deg": block_size_deg,
            "train_years": train_years,
            "test_year": test_year,
            "val_ratio": val_ratio,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_split(
    df: pd.DataFrame,
    result: SplitResult,
) -> list[str]:
    """Run all leakage checks on a split result.

    Returns a list of error messages (empty = all checks pass).

    Checks performed (varies by protocol):

    ===== ============================================================
    All   No sample_id overlap across train/val/test
    All   All sample_ids exist in df
    Spatial No grid_id overlap between train and test
    Temporal No target_year in test appears in train
    Region Train/test regions disjoint
    Climate Train/test climate types disjoint
    Spatiotemp Both spatial AND temporal constraints hold
    ===== ============================================================
    """
    errors: list[str] = []

    train_set = set(result.train_ids)
    val_set = set(result.val_ids)
    test_set = set(result.test_ids)

    # --- universal checks ---
    errors.extend(_check_no_id_overlap(train_set, val_set, test_set))
    errors.extend(_check_ids_exist(df, result))

    # --- protocol-specific checks ---
    if result.protocol in ("spatial_block", "spatial", "spatial_holdout"):
        errors.extend(_check_spatial_disjoint(df, result))
    elif result.protocol == "temporal":
        errors.extend(_check_temporal_disjoint(df, result))
    elif result.protocol == "region_transfer":
        errors.extend(_check_region_disjoint(df, result))
    elif result.protocol == "climate_zone_transfer":
        errors.extend(_check_climate_zone_disjoint(df, result))
    elif result.protocol == "spatiotemporal":
        errors.extend(_check_spatial_disjoint(df, result))
        errors.extend(_check_temporal_disjoint(df, result))

    return errors


def _check_no_id_overlap(
    train: set[str], val: set[str], test: set[str]
) -> list[str]:
    errors: list[str] = []
    if train & test:
        errors.append(f"ID overlap between train and test: {len(train & test)} samples")
    if train & val:
        errors.append(f"ID overlap between train and val: {len(train & val)} samples")
    if val & test:
        errors.append(f"ID overlap between val and test: {len(val & test)} samples")
    return errors


def _check_ids_exist(df: pd.DataFrame, result: SplitResult) -> list[str]:
    errors: list[str] = []
    all_df_ids = set(df["sample_id"])
    for label, ids in [
        ("train", result.train_ids),
        ("val", result.val_ids),
        ("test", result.test_ids),
    ]:
        missing = set(ids) - all_df_ids
        if missing:
            errors.append(
                f"{label} contains {len(missing)} sample_ids not in DataFrame"
            )
    return errors


def _check_spatial_disjoint(df: pd.DataFrame, result: SplitResult) -> list[str]:
    errors: list[str] = []
    train_grids = set(df[df["sample_id"].isin(result.train_ids)]["grid_id"].unique())
    test_grids = set(df[df["sample_id"].isin(result.test_ids)]["grid_id"].unique())
    overlap = train_grids & test_grids
    if overlap:
        errors.append(
            f"Spatial leakage: {len(overlap)} grid_ids appear in both "
            f"train and test"
        )
    return errors


def _check_temporal_disjoint(df: pd.DataFrame, result: SplitResult) -> list[str]:
    errors: list[str] = []
    train_years = set(df[df["sample_id"].isin(result.train_ids)]["target_year"].unique())
    test_years = set(df[df["sample_id"].isin(result.test_ids)]["target_year"].unique())
    overlap = train_years & test_years
    if overlap:
        errors.append(
            f"Temporal leakage: years {sorted(overlap)} appear in both "
            f"train and test"
        )
    return errors


def _check_region_disjoint(df: pd.DataFrame, result: SplitResult) -> list[str]:
    errors: list[str] = []
    train_regions = set(df[df["sample_id"].isin(result.train_ids)]["region"].unique())
    test_regions = set(df[df["sample_id"].isin(result.test_ids)]["region"].unique())
    overlap = train_regions & test_regions
    if overlap:
        errors.append(
            f"Region leakage: {sorted(overlap)} appear in both train and test"
        )
    return errors


def _check_climate_zone_disjoint(df: pd.DataFrame, result: SplitResult) -> list[str]:
    errors: list[str] = []
    train_zones = set(
        df[df["sample_id"].isin(result.train_ids)]["climate_type"].unique()
    )
    test_zones = set(
        df[df["sample_id"].isin(result.test_ids)]["climate_type"].unique()
    )
    overlap = train_zones & test_zones
    if overlap:
        errors.append(
            f"Climate-zone leakage: {sorted(overlap)} appear in both "
            f"train and test"
        )
    return errors


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_split_result(
    result: SplitResult,
    output_dir: str | Path,
) -> Path:
    """Save a split result to disk.

    Creates::

        {output_dir}/
        ├── train_ids.csv
        ├── val_ids.csv
        ├── test_ids.csv
        └── split_metadata.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for label, ids in [
        ("train", result.train_ids),
        ("val", result.val_ids),
        ("test", result.test_ids),
    ]:
        pd.DataFrame({"sample_id": ids}).to_csv(out / f"{label}_ids.csv", index=False)

    meta = {
        "split_id": result.split_id,
        "protocol": result.protocol,
        "config": result.config,
        **result.metadata,
    }
    with (out / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return out


def load_split_result(split_dir: str | Path) -> SplitResult:
    """Load a previously saved split result from disk."""
    d = Path(split_dir)
    with (d / "split_metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    result = SplitResult(
        split_id=meta.get("split_id", d.name),
        protocol=meta.get("protocol", "unknown"),
        config=meta.get("config", {}),
        metadata={k: v for k, v in meta.items() if k not in ("split_id", "protocol", "config")},
    )

    for label in ("train", "val", "test"):
        path = d / f"{label}_ids.csv"
        if path.exists():
            ids = pd.read_csv(path)["sample_id"].tolist()
            setattr(result, f"{label}_ids", ids)

    return result


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate_all_splits(
    df: pd.DataFrame,
    output_root: str | Path,
    configs: dict[str, dict[str, Any]] | None = None,
) -> list[SplitResult]:
    """Generate all six benchmark split protocols and save to disk.

    Parameters
    ----------
    df
        Forecasting samples DataFrame (must have ``sample_id``, ``grid_id``,
        ``region``, ``climate_type``, ``target_year``, ``latitude``,
        ``longitude``).
    output_root
        Root directory for split outputs (e.g. ``outputs/benchmark/splits``).
    configs
        Optional per-protocol config dict keyed by protocol name.
        If not provided, sensible defaults are used.

    Returns
    -------
    List of :class:`SplitResult` objects, one per protocol.
    """
    if configs is None:
        configs = _default_split_configs(df)
    else:
        configs = {**_default_split_configs(df), **configs}

    root = Path(output_root)
    results: list[SplitResult] = []

    # 1 — random
    cfg = configs.get("random", {})
    r = make_random_split(df, **{k: v for k, v in cfg.items() if k != "split_id"})
    save_split_result(r, root / r.split_id)
    results.append(r)

    # 2 — spatial_block
    cfg = configs.get("spatial_block", {})
    r = make_spatial_block_split(df, **{k: v for k, v in cfg.items() if k != "split_id"})
    save_split_result(r, root / r.split_id)
    results.append(r)

    # 3 — temporal
    cfg = configs.get("temporal", {})
    r = make_temporal_split(df, **{k: v for k, v in cfg.items() if k != "split_id"})
    save_split_result(r, root / r.split_id)
    results.append(r)

    # 4 — region_transfer (one per pair)
    rt_cfg = configs.get("region_transfer", {})
    pairs = rt_cfg.pop("pairs", None)
    if pairs is None:
        pairs = _default_region_pairs(df)
    for i, pair in enumerate(pairs):
        sp_id = f"region_transfer_{i}"
        r = make_region_transfer_split(
            df,
            train_regions=pair["train_regions"],
            test_regions=pair["test_regions"],
            **{k: v for k, v in rt_cfg.items() if k not in ("pairs", "split_id")},
            split_id=sp_id,
        )
        save_split_result(r, root / r.split_id)
        results.append(r)

    # 5 — climate_zone_transfer (one per pair)
    cz_cfg = configs.get("climate_zone_transfer", {})
    cz_pairs = cz_cfg.pop("pairs", None)
    if cz_pairs is None:
        cz_pairs = _default_climate_zone_pairs(df)
    for i, pair in enumerate(cz_pairs):
        sp_id = f"climate_zone_transfer_{i}"
        r = make_climate_zone_transfer_split(
            df,
            train_zones=pair["train_zones"],
            test_zones=pair["test_zones"],
            **{k: v for k, v in cz_cfg.items() if k not in ("pairs", "split_id")},
            split_id=sp_id,
        )
        save_split_result(r, root / r.split_id)
        results.append(r)

    # 6 — spatiotemporal
    cfg = configs.get("spatiotemporal", {})
    r = make_spatiotemporal_split(df, **{k: v for k, v in cfg.items() if k != "split_id"})
    save_split_result(r, root / r.split_id)
    results.append(r)

    return results


def _default_split_configs(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build sensible default split configs from the DataFrame's contents."""
    years = sorted(df["target_year"].unique().tolist())
    if len(years) >= 4:
        train_years = years[:-2]
        val_year = years[-2]
        test_year = years[-1]
    elif len(years) == 3:
        train_years = years[:1]
        val_year = years[1]
        test_year = years[2]
    else:
        train_years = years[:1]
        val_year = None
        test_year = years[-1]

    return {
        "random": {"split_id": "random", "seed": RANDOM_SEED},
        "spatial_block": {
            "split_id": "spatial_block",
            "block_size_deg": 5.0,
            "seed": RANDOM_SEED,
        },
        "temporal": {
            "split_id": "temporal",
            "train_years": train_years,
            "val_year": val_year,
            "test_year": test_year,
            "seed": RANDOM_SEED,
        },
        "region_transfer": {
            "split_id": "region_transfer",
            "pairs": _default_region_pairs(df),
            "seed": RANDOM_SEED,
        },
        "climate_zone_transfer": {
            "split_id": "climate_zone_transfer",
            "pairs": _default_climate_zone_pairs(df),
            "seed": RANDOM_SEED,
        },
        "spatiotemporal": {
            "split_id": "spatiotemporal",
            "block_size_deg": 5.0,
            "train_years": train_years,
            "test_year": test_year,
            "seed": RANDOM_SEED,
        },
    }


def _default_region_pairs(df: pd.DataFrame) -> list[dict[str, list[str]]]:
    regions = sorted(df["region"].unique().tolist())
    if len(regions) < 2:
        return []
    pairs: list[dict[str, list[str]]] = []
    for held_out in regions:
        train = [r for r in regions if r != held_out]
        pairs.append({"train_regions": train, "test_regions": [held_out]})
    return pairs


def _default_climate_zone_pairs(df: pd.DataFrame) -> list[dict[str, list[str]]]:
    zones = sorted(df["climate_type"].unique().tolist())
    if len(zones) < 2:
        return []
    pairs: list[dict[str, list[str]]] = []
    for held_out in zones:
        train = [z for z in zones if z != held_out]
        pairs.append({"train_zones": train, "test_zones": [held_out]})
    return pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Train/val/test ratios must sum to 1.0, got {train} + {val} + {test} = {total}"
        )
