"""Tests for benchmark split protocols."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.benchmark.split_protocols import (
    RANDOM_SEED,
    SplitResult,
    _make_spatial_block_id,
    generate_all_splits,
    load_split_result,
    make_climate_zone_transfer_split,
    make_random_split,
    make_region_transfer_split,
    make_spatial_block_split,
    make_spatiotemporal_split,
    make_temporal_split,
    save_split_result,
    validate_split,
)


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------


def make_test_forecasting_samples(
    n_regions: int = 3,
    n_cells_per_region: int = 4,
    n_years: int = 4,
    months_per_year: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a small but realistic forecasting sample table for testing.

    Creates ``n_regions`` × ``n_cells_per_region`` grid cells ×
    ``n_years`` years × ``months_per_year`` months.

    Each cell starts with month 7 so the first target is in year[0]-month[7],
    giving a natural 6-month gap at the start that mirrors real forecasting
    data (where the first 6 months of each cell are used as history).
    """
    rng = np.random.default_rng(seed)
    regions_list = ["Sahara", "East China", "Amazon", "Central Europe", "Western US"]
    climate_map = {
        "Sahara": "arid",
        "East China": "monsoon",
        "Amazon": "tropical_humid",
        "Central Europe": "temperate",
        "Western US": "semi_arid",
    }

    rows = []
    start_year = 2019
    for ri, region in enumerate(regions_list[:n_regions]):
        for ci in range(n_cells_per_region):
            lat = 20.0 + ri * 10.0 + ci * 0.5
            lon = 10.0 + ri * 30.0 + ci * 1.0
            grid_id = f"{lat:.4f}_{lon:.4f}"
            sample_counter = 0
            for yi in range(n_years):
                year = start_year + yi
                for month in range(1, months_per_year + 1):
                    sample_counter += 1
                    sample_id = f"{region}_{grid_id}_{year}_{month:02d}"
                    # For the first 6 months, no valid sample exists in real data
                    # (they serve as history). For tests, we include them to
                    # verify incomplete history handling where relevant.

                    rows.append(
                        {
                            "sample_id": sample_id,
                            "grid_id": grid_id,
                            "region": region,
                            "climate_type": climate_map[region],
                            "target_year": year,
                            "target_month": month,
                            "latitude": lat,
                            "longitude": lon,
                            "input_window_start": f"{year}-{max(1, month-6):02d}",
                            "input_window_end": f"{year}-{max(1, month-1):02d}",
                            "y_true": float(sample_counter) + rng.normal(0, 0.1),
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spatial block ID helper
# ---------------------------------------------------------------------------


class TestSpatialBlockId:
    def test_block_id_5deg(self) -> None:
        bid = _make_spatial_block_id(23.4, 115.8, 5.0)
        assert bid == "block_lat20_lon115"

    def test_block_id_negative(self) -> None:
        bid = _make_spatial_block_id(-3.5, -62.1, 5.0)
        assert bid == "block_lat-5_lon-65"

    def test_block_id_different_sizes(self) -> None:
        bid = _make_spatial_block_id(23.4, 115.8, 10.0)
        assert bid == "block_lat20_lon110"


# ---------------------------------------------------------------------------
# Protocol 1 — Random split
# ---------------------------------------------------------------------------


class TestRandomSplit:
    def test_basic_split(self) -> None:
        df = make_test_forecasting_samples()
        r = make_random_split(df)
        assert r.protocol == "random"
        assert len(r.train_ids) > 0
        assert len(r.val_ids) > 0
        assert len(r.test_ids) > 0
        # No overlap
        assert set(r.train_ids) & set(r.test_ids) == set()

    def test_deterministic_with_seed(self) -> None:
        df = make_test_forecasting_samples()
        r1 = make_random_split(df, seed=42)
        r2 = make_random_split(df, seed=42)
        assert r1.train_ids == r2.train_ids
        assert r1.test_ids == r2.test_ids

    def test_different_seed_gives_different_split(self) -> None:
        df = make_test_forecasting_samples()
        r1 = make_random_split(df, seed=42)
        r2 = make_random_split(df, seed=99)
        assert r1.train_ids != r2.train_ids

    def test_ratios_approximately_correct(self) -> None:
        df = make_test_forecasting_samples(n_regions=2, n_cells_per_region=5, n_years=5)
        r = make_random_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        n = len(df)
        assert abs(len(r.train_ids) / n - 0.7) < 0.05
        assert abs(len(r.test_ids) / n - 0.15) < 0.05

    def test_ratios_must_sum_to_one(self) -> None:
        df = make_test_forecasting_samples()
        with pytest.raises(ValueError, match="sum to 1"):
            make_random_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.3)


# ---------------------------------------------------------------------------
# Protocol 2 — Spatial block holdout
# ---------------------------------------------------------------------------


class TestSpatialBlockSplit:
    def test_basic_split(self) -> None:
        df = make_test_forecasting_samples(n_regions=2, n_cells_per_region=10, n_years=3)
        r = make_spatial_block_split(df, block_size_deg=5.0)
        assert r.protocol == "spatial_block"
        assert len(r.train_ids) > 0
        assert len(r.test_ids) > 0

    def test_no_grid_id_overlap(self) -> None:
        df = make_test_forecasting_samples(n_regions=2, n_cells_per_region=10, n_years=3)
        r = make_spatial_block_split(df, block_size_deg=5.0)
        train_grids = set(df[df["sample_id"].isin(r.train_ids)]["grid_id"].unique())
        test_grids = set(df[df["sample_id"].isin(r.test_ids)]["grid_id"].unique())
        assert not train_grids & test_grids

    def test_too_few_blocks_raises(self) -> None:
        df = make_test_forecasting_samples(n_regions=1, n_cells_per_region=2, n_years=1)
        with pytest.raises(ValueError, match="at least 3 spatial blocks"):
            make_spatial_block_split(df, block_size_deg=100.0)

    def test_deterministic(self) -> None:
        df = make_test_forecasting_samples()
        r1 = make_spatial_block_split(df, seed=42)
        r2 = make_spatial_block_split(df, seed=42)
        assert r1.train_ids == r2.train_ids


# ---------------------------------------------------------------------------
# Protocol 3 — Temporal holdout
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def test_basic_split(self) -> None:
        df = make_test_forecasting_samples(n_regions=2, n_years=4)
        r = make_temporal_split(
            df, train_years=[2019, 2020], val_year=2021, test_year=2022
        )
        assert r.protocol == "temporal"
        assert len(r.train_ids) > 0
        assert len(r.test_ids) > 0

    def test_no_year_leakage(self) -> None:
        df = make_test_forecasting_samples(n_regions=2, n_years=4)
        r = make_temporal_split(
            df, train_years=[2019, 2020], val_year=2021, test_year=2022
        )
        train_years = set(df[df["sample_id"].isin(r.train_ids)]["target_year"])
        test_years = set(df[df["sample_id"].isin(r.test_ids)]["target_year"])
        assert 2022 not in train_years
        assert 2022 in test_years

    def test_future_years_not_in_train(self) -> None:
        df = make_test_forecasting_samples(n_regions=2, n_years=4)
        r = make_temporal_split(
            df, train_years=[2019, 2020], test_year=2022
        )
        train_years = set(df[df["sample_id"].isin(r.train_ids)]["target_year"])
        assert max(train_years) < 2022

    def test_no_samples_for_train_years_raises(self) -> None:
        df = make_test_forecasting_samples(n_years=2)
        with pytest.raises(ValueError, match="No samples found"):
            make_temporal_split(df, train_years=[2050, 2051], test_year=2052)


# ---------------------------------------------------------------------------
# Protocol 4 — Region transfer
# ---------------------------------------------------------------------------


class TestRegionTransferSplit:
    def test_basic_split(self) -> None:
        df = make_test_forecasting_samples(n_regions=3)
        r = make_region_transfer_split(
            df,
            train_regions=["Sahara", "East China"],
            test_regions=["Amazon"],
        )
        assert r.protocol == "region_transfer"
        assert len(r.train_ids) > 0
        assert len(r.test_ids) > 0

    def test_regions_disjoint(self) -> None:
        df = make_test_forecasting_samples(n_regions=3)
        r = make_region_transfer_split(
            df,
            train_regions=["Sahara", "East China"],
            test_regions=["Amazon"],
        )
        train_regions = set(df[df["sample_id"].isin(r.train_ids)]["region"])
        test_regions = set(df[df["sample_id"].isin(r.test_ids)]["region"])
        assert "Amazon" not in train_regions
        assert "Amazon" in test_regions

    def test_overlap_raises(self) -> None:
        df = make_test_forecasting_samples(n_regions=3)
        with pytest.raises(ValueError, match="overlap"):
            make_region_transfer_split(
                df,
                train_regions=["Sahara", "East China"],
                test_regions=["Sahara"],
            )

    def test_empty_train_raises(self) -> None:
        df = make_test_forecasting_samples(n_regions=1)
        with pytest.raises(ValueError, match="No samples found"):
            make_region_transfer_split(
                df,
                train_regions=["Mars"],
                test_regions=["Sahara"],
            )


# ---------------------------------------------------------------------------
# Protocol 5 — Climate-zone transfer
# ---------------------------------------------------------------------------


class TestClimateZoneTransferSplit:
    def test_basic_split(self) -> None:
        df = make_test_forecasting_samples(n_regions=3)
        # Sahara=arid, East China=monsoon, Amazon=tropical_humid
        r = make_climate_zone_transfer_split(
            df,
            train_zones=["arid", "monsoon"],
            test_zones=["tropical_humid"],
        )
        assert r.protocol == "climate_zone_transfer"
        assert len(r.train_ids) > 0
        assert len(r.test_ids) > 0

    def test_zones_disjoint(self) -> None:
        df = make_test_forecasting_samples(n_regions=3)
        r = make_climate_zone_transfer_split(
            df,
            train_zones=["arid", "monsoon"],
            test_zones=["tropical_humid"],
        )
        train_zones = set(df[df["sample_id"].isin(r.train_ids)]["climate_type"])
        test_zones = set(df[df["sample_id"].isin(r.test_ids)]["climate_type"])
        assert "tropical_humid" not in train_zones
        assert "tropical_humid" in test_zones

    def test_overlap_raises(self) -> None:
        df = make_test_forecasting_samples(n_regions=2)
        with pytest.raises(ValueError, match="overlap"):
            make_climate_zone_transfer_split(
                df,
                train_zones=["arid", "monsoon"],
                test_zones=["arid"],
            )

    def test_missing_climate_type_column_raises(self) -> None:
        df = make_test_forecasting_samples(n_regions=2)
        df.drop(columns=["climate_type"], inplace=True)
        with pytest.raises(ValueError, match="climate_type"):
            make_climate_zone_transfer_split(
                df,
                train_zones=["arid"],
                test_zones=["monsoon"],
            )


# ---------------------------------------------------------------------------
# Protocol 6 — Spatiotemporal holdout
# ---------------------------------------------------------------------------


class TestSpatiotemporalSplit:
    def test_basic_split(self) -> None:
        df = make_test_forecasting_samples(
            n_regions=2, n_cells_per_region=12, n_years=4
        )
        # Use years that actually exist in the test data (2019–2022).
        r = make_spatiotemporal_split(
            df,
            block_size_deg=5.0,
            train_years=[2019, 2020],
            test_year=2022,
        )
        assert r.protocol == "spatiotemporal"
        assert len(r.train_ids) > 0
        assert len(r.test_ids) > 0

    def test_both_constraints_hold(self) -> None:
        df = make_test_forecasting_samples(
            n_regions=2, n_cells_per_region=12, n_years=4
        )
        r = make_spatiotemporal_split(
            df,
            block_size_deg=5.0,
            train_years=[2019, 2020],
            test_year=2022,
        )
        # Spatial: no grid_id overlap
        train_grids = set(df[df["sample_id"].isin(r.train_ids)]["grid_id"].unique())
        test_grids = set(df[df["sample_id"].isin(r.test_ids)]["grid_id"].unique())
        assert not train_grids & test_grids
        # Temporal: test year not in train
        train_years = set(df[df["sample_id"].isin(r.train_ids)]["target_year"])
        assert 2022 not in train_years

    def test_empty_split_raises(self) -> None:
        """Using a train_years list with no data should raise."""
        df = make_test_forecasting_samples(n_regions=1, n_cells_per_region=2, n_years=2)
        # Data has years 2019–2020.  Asking for 2050 with small blocks
        # (so each cell gets its own block) should produce empty train.
        with pytest.raises(ValueError, match="empty"):
            make_spatiotemporal_split(
                df,
                block_size_deg=1.0,
                train_years=[2050],
                test_year=2051,
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateSplit:
    def test_all_protocols_pass_validation(self) -> None:
        df = make_test_forecasting_samples(
            n_regions=3, n_cells_per_region=12, n_years=4
        )

        r1 = make_random_split(df)
        assert validate_split(df, r1) == []

        r2 = make_spatial_block_split(df, block_size_deg=5.0)
        assert validate_split(df, r2) == []

        r3 = make_temporal_split(
            df, train_years=[2019, 2020], val_year=2021, test_year=2022
        )
        assert validate_split(df, r3) == []

        r4 = make_region_transfer_split(
            df,
            train_regions=["Sahara", "East China"],
            test_regions=["Amazon"],
        )
        assert validate_split(df, r4) == []

        r5 = make_climate_zone_transfer_split(
            df,
            train_zones=["arid", "monsoon"],
            test_zones=["tropical_humid"],
        )
        assert validate_split(df, r5) == []

        r6 = make_spatiotemporal_split(
            df,
            block_size_deg=5.0,
            train_years=[2019, 2020],
            test_year=2022,
        )
        assert validate_split(df, r6) == []

    def test_detects_id_overlap(self) -> None:
        df = make_test_forecasting_samples()
        r = make_random_split(df)
        # Artificially create overlap
        r.train_ids.append(r.test_ids[0])
        errors = validate_split(df, r)
        assert any("overlap" in e for e in errors)

    def test_detects_nonexistent_ids(self) -> None:
        df = make_test_forecasting_samples()
        r = make_random_split(df)
        r.train_ids.append("nonexistent_sample_999")
        errors = validate_split(df, r)
        assert any("not in DataFrame" in e for e in errors)

    def test_detects_spatial_leakage(self) -> None:
        df = make_test_forecasting_samples(n_regions=2, n_cells_per_region=8, n_years=3)
        r = make_spatial_block_split(df, block_size_deg=5.0)
        # Move a test sample into train
        if r.test_ids:
            r.train_ids.append(r.test_ids[0])
            errors = validate_split(df, r)
            assert any("overlap" in e for e in errors)

    def test_detects_temporal_leakage(self) -> None:
        df = make_test_forecasting_samples(n_years=4)
        r = make_temporal_split(
            df, train_years=[2019, 2020], test_year=2022
        )
        # Artificially add a test-year sample to train
        test_sample = r.test_ids[0] if r.test_ids else None
        if test_sample:
            r.train_ids.append(test_sample)
            errors = validate_split(df, r)
            assert any("overlap" in e for e in errors) or any(
                "Temporal" in e for e in errors
            )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


class TestSplitIO:
    def test_save_and_load_roundtrip(self) -> None:
        df = make_test_forecasting_samples()
        r = make_random_split(df)

        with tempfile.TemporaryDirectory() as tmp:
            save_split_result(r, tmp)
            loaded = load_split_result(tmp)

            assert loaded.split_id == r.split_id
            assert loaded.protocol == r.protocol
            assert loaded.train_ids == r.train_ids
            assert loaded.val_ids == r.val_ids
            assert loaded.test_ids == r.test_ids

    def test_generate_all_splits(self) -> None:
        df = make_test_forecasting_samples(
            n_regions=3, n_cells_per_region=10, n_years=4
        )
        with tempfile.TemporaryDirectory() as tmp:
            results = generate_all_splits(df, tmp)
            # At minimum: random, spatial_block, temporal, spatiotemporal
            # + N region_transfer pairs + N climate_zone_transfer pairs
            n_expected = 4  # random, spatial_block, temporal, spatiotemporal
            n_regions = df["region"].nunique()
            n_zones = df["climate_type"].nunique()
            n_expected += n_regions + n_zones  # one per held-out region/zone
            assert len(results) >= n_expected

            # Verify all split dirs exist
            for result in results:
                split_dir = Path(tmp) / result.split_id
                assert split_dir.exists()
                assert (split_dir / "train_ids.csv").exists()
                assert (split_dir / "split_metadata.json").exists()

                # All IDs should exist in df
                all_ids = set(result.train_ids + result.val_ids + result.test_ids)
                df_ids = set(df["sample_id"])
                missing = all_ids - df_ids
                assert not missing, f"{result.split_id}: {len(missing)} IDs not in df"
