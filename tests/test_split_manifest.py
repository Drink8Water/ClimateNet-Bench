"""Tests for fixed split-manifest generation and audit.

Covers the four split types:

1. random
2. temporal_holdout
3. climate_zone_holdout
4. spatial_temporal_holdout

Every test verifies:
- No sample_id overlap across train/val/test.
- All three sets are non-empty.
- The CSV files are written to disk.
- The audit function detects violations correctly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.benchmark.split_manifest import (
    RANDOM_SEED,
    audit_split_manifest,
    generate_climate_zone_holdout_split,
    generate_random_split,
    generate_spatial_temporal_holdout_split,
    generate_temporal_holdout_split,
)


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------


def _make_synth_samples(
    n_regions: int = 3,
    n_years: int = 4,
    points_per_region: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a small forecasting-sample-like DataFrame for split testing.

    Returns a DataFrame with columns:
    sample_id, year, month, lat, lon, region, climate_zone
    """
    rng = np.random.default_rng(seed)
    regions = [
        ("Sahara", "arid", (22.0, 10.0)),
        ("East China", "monsoon", (30.0, 115.0)),
        ("Amazon", "tropical_humid", (-3.0, -60.0)),
        ("Central Europe", "temperate", (50.0, 10.0)),
        ("Western US", "semi_arid", (38.0, -120.0)),
    ]
    rows = []
    for ri in range(n_regions):
        region_name, climate_zone, (base_lat, base_lon) = regions[ri]
        for pi in range(points_per_region):
            lat = base_lat + pi * 0.3
            lon = base_lon + pi * 0.3
            for yi in range(n_years):
                year = 2020 + yi
                for month in range(1, 13):
                    sample_id = f"{region_name}_{pi}_{year}_{month:02d}"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "year": year,
                            "month": month,
                            "lat": round(lat, 4),
                            "lon": round(lon, 4),
                            "region": region_name,
                            "climate_zone": climate_zone,
                            "y_true": rng.normal(0, 1),
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _assert_no_overlap(train_df, val_df, test_df):
    """All three sets must have disjoint sample_id values."""
    t_ids = set(train_df["sample_id"])
    v_ids = set(val_df["sample_id"])
    ts_ids = set(test_df["sample_id"])
    assert not (t_ids & v_ids), "Overlap between train and val"
    assert not (t_ids & ts_ids), "Overlap between train and test"
    assert not (v_ids & ts_ids), "Overlap between val and test"


def _assert_all_nonempty(train_df, val_df, test_df):
    assert len(train_df) > 0, "train is empty"
    assert len(val_df) > 0, "val is empty"
    assert len(test_df) > 0, "test is empty"


def _assert_files_exist(output_dir: Path):
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "val.csv").exists()
    assert (output_dir / "test.csv").exists()
    assert (output_dir / "split_metadata.json").exists()


def _assert_split_column(train_df, val_df, test_df):
    assert (train_df["split"] == "train").all()
    assert (val_df["split"] == "val").all()
    assert (test_df["split"] == "test").all()


# ---------------------------------------------------------------------------
# 1. Random split
# ---------------------------------------------------------------------------


class TestRandomSplit:
    def test_no_overlap_and_nonempty(self) -> None:
        df = _make_synth_samples()
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_random_split(df, tmp, seed=42)
            _assert_files_exist(Path(tmp))

            train = pd.read_csv(Path(tmp) / "train.csv")
            val = pd.read_csv(Path(tmp) / "val.csv")
            test = pd.read_csv(Path(tmp) / "test.csv")

            _assert_no_overlap(train, val, test)
            _assert_all_nonempty(train, val, test)
            _assert_split_column(train, val, test)

            assert result["audit"]["sample_overlap"] == "PASS"
            assert result["metadata"]["split_type"] == "random"

    def test_fractions_approx_correct(self) -> None:
        df = _make_synth_samples(n_regions=3, n_years=5, points_per_region=4)
        n = len(df)
        with tempfile.TemporaryDirectory() as tmp:
            generate_random_split(df, tmp, train_frac=0.70, val_frac=0.15, test_frac=0.15)
            train = pd.read_csv(Path(tmp) / "train.csv")
            val = pd.read_csv(Path(tmp) / "val.csv")
            test = pd.read_csv(Path(tmp) / "test.csv")
            assert abs(len(train) / n - 0.70) < 0.03
            assert abs(len(val) / n - 0.15) < 0.03
            assert abs(len(test) / n - 0.15) < 0.03

    def test_deterministic(self) -> None:
        df = _make_synth_samples()
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            generate_random_split(df, t1, seed=42)
            generate_random_split(df, t2, seed=42)
            tr1 = pd.read_csv(Path(t1) / "train.csv")
            tr2 = pd.read_csv(Path(t2) / "train.csv")
            assert tr1["sample_id"].tolist() == tr2["sample_id"].tolist()

    def test_raises_on_invalid_fractions(self) -> None:
        df = _make_synth_samples()
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="must equal 1.0"):
                generate_random_split(df, tmp, train_frac=0.5, val_frac=0.3, test_frac=0.3)

    def test_audit_passes(self) -> None:
        df = _make_synth_samples()
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_random_split(df, tmp)
            audit = result["audit"]
            assert audit["sample_overlap"] == "PASS"
            assert audit["train_size"] > 0
            assert audit["val_size"] > 0
            assert audit["test_size"] > 0


# ---------------------------------------------------------------------------
# 2. Temporal holdout
# ---------------------------------------------------------------------------


class TestTemporalHoldout:
    def test_no_year_leakage(self) -> None:
        df = _make_synth_samples(n_years=5)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_temporal_holdout_split(
                df, tmp,
                train_years=[2020, 2021],
                val_years=[2022],
                test_years=[2023, 2024],
            )
            train = pd.read_csv(Path(tmp) / "train.csv")
            val = pd.read_csv(Path(tmp) / "val.csv")
            test = pd.read_csv(Path(tmp) / "test.csv")

            train_years_set = set(train["year"].unique())
            test_years_set = set(test["year"].unique())

            assert 2023 not in train_years_set
            assert 2024 not in train_years_set
            assert not (train_years_set & test_years_set)

            _assert_no_overlap(train, val, test)
            _assert_all_nonempty(train, val, test)

            assert result["audit"]["temporal_overlap"] == "PASS"

    def test_auto_assign_years(self) -> None:
        df = _make_synth_samples(n_years=6)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_temporal_holdout_split(df, tmp)
            train = pd.read_csv(Path(tmp) / "train.csv")
            test = pd.read_csv(Path(tmp) / "test.csv")
            train_years = set(train["year"].unique())
            test_years = set(test["year"].unique())
            assert not (train_years & test_years)

            audit = result["audit"]
            assert audit["temporal_overlap"] == "PASS"

    def test_raises_when_test_year_in_train(self) -> None:
        df = _make_synth_samples(n_years=4)
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Temporal leakage"):
                generate_temporal_holdout_split(
                    df, tmp,
                    train_years=[2020, 2021, 2022],
                    val_years=[2021],
                    test_years=[2022],
                )

    def test_not_enough_years_raises(self) -> None:
        df = _make_synth_samples(n_years=1)
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="at least 3 unique years"):
                generate_temporal_holdout_split(df, tmp)


# ---------------------------------------------------------------------------
# 3. Climate-zone holdout
# ---------------------------------------------------------------------------


class TestClimateZoneHoldout:
    def test_no_zone_leakage(self) -> None:
        df = _make_synth_samples(n_regions=4)  # arid, monsoon, tropical_humid, temperate
        zones = sorted(df["climate_zone"].unique())
        train_zones = zones[:2]
        val_zones = [zones[2]]
        test_zones = [zones[3]]

        with tempfile.TemporaryDirectory() as tmp:
            result = generate_climate_zone_holdout_split(
                df, tmp,
                train_zones=train_zones,
                val_zones=val_zones,
                test_zones=test_zones,
            )
            train = pd.read_csv(Path(tmp) / "train.csv")
            test = pd.read_csv(Path(tmp) / "test.csv")

            train_zone_set = set(train["climate_zone"].unique())
            test_zone_set = set(test["climate_zone"].unique())

            assert not (train_zone_set & test_zone_set)

            _assert_no_overlap(train, pd.read_csv(Path(tmp) / "val.csv"), test)
            _assert_all_nonempty(train, pd.read_csv(Path(tmp) / "val.csv"), test)

            assert result["audit"]["climate_zone_overlap"] == "PASS"

    def test_auto_assign_zones(self) -> None:
        df = _make_synth_samples(n_regions=4)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_climate_zone_holdout_split(df, tmp)
            train = pd.read_csv(Path(tmp) / "train.csv")
            test = pd.read_csv(Path(tmp) / "test.csv")
            assert not (set(train["climate_zone"]) & set(test["climate_zone"]))

            audit = result["audit"]
            assert audit["climate_zone_overlap"] == "PASS"

    def test_raises_when_test_zone_in_train(self) -> None:
        df = _make_synth_samples(n_regions=3)
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Climate-zone leakage"):
                generate_climate_zone_holdout_split(
                    df, tmp,
                    train_zones=["arid", "monsoon"],
                    val_zones=["arid"],
                    test_zones=["arid"],
                )

    def test_not_enough_zones_raises(self) -> None:
        df = _make_synth_samples(n_regions=1)
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="at least 2 unique climate zones"):
                generate_climate_zone_holdout_split(df, tmp)


# ---------------------------------------------------------------------------
# 4. Spatial-temporal holdout
# ---------------------------------------------------------------------------


class TestSpatialTemporalHoldout:
    def test_no_region_or_year_leakage(self) -> None:
        df = _make_synth_samples(n_regions=4, n_years=6)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_spatial_temporal_holdout_split(
                df, tmp,
                train_regions=["Sahara", "East China"],
                val_regions=["Amazon"],
                test_regions=["Central Europe"],
                train_years=[2020, 2021, 2022],
                val_years=[2023],
                test_years=[2024, 2025],
            )
            train = pd.read_csv(Path(tmp) / "train.csv")
            val = pd.read_csv(Path(tmp) / "val.csv")
            test = pd.read_csv(Path(tmp) / "test.csv")

            train_regions_set = set(train["region"].unique())
            test_regions_set = set(test["region"].unique())
            train_years_set = set(train["year"].unique())
            test_years_set = set(test["year"].unique())

            assert not (train_regions_set & test_regions_set), "spatial leakage"
            assert not (train_years_set & test_years_set), "temporal leakage"

            _assert_no_overlap(train, val, test)
            _assert_all_nonempty(train, val, test)

            assert result["audit"]["temporal_overlap"] == "PASS"
            assert result["audit"]["spatial_overlap"] == "PASS"

    def test_auto_assign(self) -> None:
        df = _make_synth_samples(n_regions=4, n_years=6)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_spatial_temporal_holdout_split(df, tmp)
            train = pd.read_csv(Path(tmp) / "train.csv")
            test = pd.read_csv(Path(tmp) / "test.csv")

            assert not (set(train["region"]) & set(test["region"]))
            assert not (set(train["year"]) & set(test["year"]))

            assert result["audit"]["temporal_overlap"] == "PASS"
            assert result["audit"]["spatial_overlap"] == "PASS"

    def test_raises_on_spatial_leakage(self) -> None:
        df = _make_synth_samples(n_regions=3, n_years=5)
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Spatial leakage"):
                generate_spatial_temporal_holdout_split(
                    df, tmp,
                    train_regions=["Sahara", "East China"],
                    test_regions=["Sahara"],
                    train_years=[2020, 2021],
                    test_years=[2023, 2024],
                )

    def test_raises_on_temporal_leakage(self) -> None:
        df = _make_synth_samples(n_regions=3, n_years=5)
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Temporal leakage"):
                generate_spatial_temporal_holdout_split(
                    df, tmp,
                    train_regions=["Sahara"],
                    test_regions=["East China"],
                    train_years=[2020, 2021, 2023],
                    test_years=[2023, 2024],
                )

    def test_raises_when_empty_split(self) -> None:
        df = _make_synth_samples(n_regions=2, n_years=2)
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError):
                generate_spatial_temporal_holdout_split(
                    df, tmp,
                    train_regions=["Sahara"],
                    test_regions=["East China"],
                    train_years=[2025],   # no data for this year
                    test_years=[2026],
                )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class TestAuditSplitManifest:
    def test_detects_sample_overlap(self) -> None:
        df = _make_synth_samples()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_random_split(df, out)

            # Corrupt: inject a train sample into test
            train = pd.read_csv(out / "train.csv")
            test = pd.read_csv(out / "test.csv")
            corrupted_test = pd.concat([test, train.iloc[[0]]], ignore_index=True)
            corrupted_test.to_csv(out / "test.csv", index=False)

            audit = audit_split_manifest(out, "random")
            assert audit["sample_overlap"] == "FAIL"
            assert any("overlap" in w.lower() for w in audit["warnings"])

    def test_detects_temporal_leakage(self) -> None:
        df = _make_synth_samples(n_years=5)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_temporal_holdout_split(
                df, out,
                train_years=[2020, 2021],
                val_years=[2022],
                test_years=[2023, 2024],
            )
            # Corrupt: add a 2023 row to train
            train = pd.read_csv(out / "train.csv")
            test = pd.read_csv(out / "test.csv")
            test_2023 = test[test["year"] == 2023].iloc[[0]].copy()
            corrupted_train = pd.concat([train, test_2023], ignore_index=True)
            corrupted_train.to_csv(out / "train.csv", index=False)

            audit = audit_split_manifest(out, "temporal_holdout")
            assert audit["temporal_overlap"] == "FAIL"

    def test_detects_climate_zone_leakage(self) -> None:
        df = _make_synth_samples(n_regions=4)
        zones = sorted(df["climate_zone"].unique())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_climate_zone_holdout_split(
                df, out,
                train_zones=zones[:2],
                val_zones=[zones[2]],
                test_zones=[zones[3]],
            )
            # Corrupt: add a test-zone row to train
            train = pd.read_csv(out / "train.csv")
            test = pd.read_csv(out / "test.csv")
            corrupted_train = pd.concat([train, test.iloc[[0]]], ignore_index=True)
            corrupted_train.to_csv(out / "train.csv", index=False)

            audit = audit_split_manifest(out, "climate_zone_holdout")
            assert audit["climate_zone_overlap"] == "FAIL"

    def test_detects_spatial_leakage(self) -> None:
        df = _make_synth_samples(n_regions=4, n_years=6)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_spatial_temporal_holdout_split(
                df, out,
                train_regions=["Sahara", "East China"],
                val_regions=["Amazon"],
                test_regions=["Central Europe"],
                train_years=[2020, 2021],
                val_years=[2022],
                test_years=[2023, 2024],
            )
            # Corrupt: add a test-region row to train
            train = pd.read_csv(out / "train.csv")
            test = pd.read_csv(out / "test.csv")
            corrupted_train = pd.concat([train, test.iloc[[0]]], ignore_index=True)
            corrupted_train.to_csv(out / "train.csv", index=False)

            audit = audit_split_manifest(out, "spatial_temporal_holdout")
            assert audit["spatial_overlap"] == "FAIL"

    def test_missing_file_returns_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = audit_split_manifest(Path(tmp), "random")
            assert audit["sample_overlap"] == "FAIL"
            assert any("Missing" in w for w in audit["warnings"])

    def test_all_required_manifest_columns_present(self) -> None:
        df = _make_synth_samples()
        expected_cols = {"sample_id", "year", "month", "lat", "lon", "region", "climate_zone", "split"}
        with tempfile.TemporaryDirectory() as tmp:
            generate_random_split(df, tmp)
            for fname in ["train.csv", "val.csv", "test.csv"]:
                manifest = pd.read_csv(Path(tmp) / fname)
                missing = expected_cols - set(manifest.columns)
                assert not missing, f"{fname} missing columns: {missing}"
