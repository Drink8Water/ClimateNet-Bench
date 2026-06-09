#!/usr/bin/env python
"""Generate all benchmark split protocols from a forecasting sample table.

Usage
-----
.. code-block:: bash

    # Default: load forecasting_samples.csv, generate all 6 protocols
    python scripts/generate_splits.py

    # Custom paths
    python scripts/generate_splits.py \\
        --input data/processed/forecasting_samples.csv \\
        --output-dir outputs/benchmark/splits

    # Generate a single protocol
    python scripts/generate_splits.py --protocol temporal

Outputs
-------
::

    outputs/benchmark/splits/
    ├── random/
    │   ├── train_ids.csv
    │   ├── val_ids.csv
    │   ├── test_ids.csv
    │   └── split_metadata.json
    ├── spatial_block/
    │   └── ...
    ├── temporal/
    │   └── ...
    ├── region_transfer_0/
    │   └── ...
    ├── climate_zone_transfer_0/
    │   └── ...
    └── spatiotemporal/
        └── ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from climatenet.benchmark.split_protocols import (
    ALL_SPLIT_NAMES,
    SplitResult,
    generate_all_splits,
    make_random_split,
    make_spatial_block_split,
    make_temporal_split,
    make_region_transfer_split,
    make_climate_zone_transfer_split,
    make_spatiotemporal_split,
    save_split_result,
    validate_split,
)
from climatenet.data.loaders import load_csv
from climatenet.utils.paths import resolve_project_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_splits")


# ---------------------------------------------------------------------------
# protocol dispatch
# ---------------------------------------------------------------------------

PROTOCOL_DISPATCH = {
    "random": make_random_split,
    "spatial_block": make_spatial_block_split,
    "temporal": make_temporal_split,
    "region_transfer": make_region_transfer_split,
    "climate_zone_transfer": make_climate_zone_transfer_split,
    "spatiotemporal": make_spatiotemporal_split,
}


def _build_synthetic_forecasting_samples() -> pd.DataFrame:
    """Build smoke-test forecasting samples via the existing script."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        samples_path = Path(tmp) / "forecasting_samples.csv"
        subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "build_forecasting_dataset.py"),
                "--synthetic",
                "--output-dir", tmp,
                "--prefix", "forecasting",
            ],
            check=True,
            capture_output=True,
        )
        return pd.read_csv(samples_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark split protocols from forecasting samples"
    )
    parser.add_argument(
        "--input",
        default="data/processed/forecasting_samples.csv",
        help="Path to forecasting_samples.csv (default: data/processed/forecasting_samples.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmark/splits",
        help="Root directory for split outputs (default: outputs/benchmark/splits)",
    )
    parser.add_argument(
        "--protocol",
        choices=ALL_SPLIT_NAMES + ["all"],
        default="all",
        help="Single protocol to generate, or 'all' for every protocol (default: all)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic demo forecasting samples instead of loading from disk",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # load forecasting samples
    # ------------------------------------------------------------------
    if args.synthetic:
        logger.info("Building synthetic forecasting samples …")
        df = _build_synthetic_forecasting_samples()
    else:
        input_path = resolve_project_path(args.input)
        logger.info("Loading forecasting samples from %s", input_path)
        df = load_csv(input_path)

    logger.info("Loaded %d samples, %d grid cells, %d regions",
                len(df), df["grid_id"].nunique(), df["region"].nunique())
    logger.info("Regions: %s", sorted(df["region"].unique()))
    logger.info("Climate types: %s", sorted(df["climate_type"].unique()))
    logger.info("Years: %s", sorted(df["target_year"].unique()))

    output_root = resolve_project_path(args.output_dir)

    # ------------------------------------------------------------------
    # generate splits
    # ------------------------------------------------------------------
    if args.protocol == "all":
        results = generate_all_splits(df, output_root)
    else:
        # Single protocol
        func = PROTOCOL_DISPATCH[args.protocol]
        # Handle region_transfer and climate_zone_transfer specially
        if args.protocol == "region_transfer":
            regions = sorted(df["region"].unique())
            if len(regions) >= 2:
                train = [r for r in regions if r != regions[-1]]
                test = [regions[-1]]
                r = func(df, train_regions=train, test_regions=test)
                save_split_result(r, output_root / r.split_id)
                results = [r]
            else:
                logger.warning("Skipping region_transfer — need >= 2 regions")
                results = []
        elif args.protocol == "climate_zone_transfer":
            zones = sorted(df["climate_type"].unique())
            if len(zones) >= 2:
                train = [z for z in zones if z != zones[-1]]
                test = [zones[-1]]
                r = func(df, train_zones=train, test_zones=test)
                save_split_result(r, output_root / r.split_id)
                results = [r]
            else:
                logger.warning("Skipping climate_zone_transfer — need >= 2 zones")
                results = []
        else:
            r = func(df)
            save_split_result(r, output_root / r.split_id)
            results = [r]

    # ------------------------------------------------------------------
    # validate and report
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  Split Protocol Results")
    print("=" * 70)

    all_ok = True
    for result in results:
        errors = validate_split(df, result)
        status = "✓ PASS" if not errors else "✗ FAIL"
        print(f"\n  [{status}] {result.split_id}")
        print(f"    Protocol:     {result.protocol}")
        print(f"    Train:        {len(result.train_ids):,}")
        print(f"    Val:          {len(result.val_ids):,}")
        print(f"    Test:         {len(result.test_ids):,}")
        if errors:
            all_ok = False
            for e in errors:
                print(f"    ERROR: {e}")
        # Show a few metadata keys
        for k, v in result.metadata.items():
            if k.startswith("n_"):
                continue  # already shown above
            if isinstance(v, list):
                print(f"    {k}: {v}")

    print()
    print(f"  Output root: {output_root}")
    print("=" * 70)

    if not all_ok:
        logger.error("Some splits failed validation — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
