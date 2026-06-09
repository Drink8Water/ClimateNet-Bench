"""Tests for the forecasting dataset constructor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.data.forecasting_dataset import (
    DEFAULT_FEATURE_COLUMNS,
    ForecastingMetadata,
    build_and_save,
    build_forecasting_samples,
    build_sequence_arrays_from_samples,
    load_forecasting_dataset,
    make_grid_id,
    make_sample_id,
    save_forecasting_dataset,
    validate_samples,
)


# ---------------------------------------------------------------------------
# Tiny synthetic test data
# ---------------------------------------------------------------------------


def make_tiny_grid_cell_data(
    n_months: int = 12,
    region: str = "Sahara",
    lat: float = 20.0,
    lon: float = 10.0,
    start_year: int = 2020,
    start_month: int = 1,
) -> pd.DataFrame:
    """Build a tiny single-grid-cell dataset with deterministic values.

    Each feature column at month m has value = m, so we can verify lag
    alignment by inspection.  The target is m * 10.
    """
    rows = []
    year = start_year
    month = start_month
    for i in range(n_months):
        # Advance year when month wraps
        current_year = start_year + (start_month + i - 1) // 12
        current_month = ((start_month + i - 1) % 12) + 1
        val = float(i + 1)  # 1, 2, 3, ...
        rows.append(
            {
                "region": region,
                "year": current_year,
                "month": current_month,
                "latitude": lat,
                "longitude": lon,
                "temperature_anomaly": val,
                "precipitation_anomaly": val + 0.1,
                "radiation_anomaly": val + 0.2,
                "soil_moisture_anomaly": val + 0.3,
                "wind_speed": val + 0.4,
                "dryness_proxy": val + 0.5,
                "saturation_vapor_pressure": val + 0.6,
                "month_sin": np.sin(2 * np.pi * current_month / 12),
                "month_cos": np.cos(2 * np.pi * current_month / 12),
                "evaporation_anomaly": val * 10.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_make_grid_id(self) -> None:
        assert make_grid_id(20.1234, -15.5678) == "20.1234_-15.5678"

    def test_make_grid_id_rounding(self) -> None:
        # 20.12351 rounds to 20.1235, -15.56784 rounds to -15.5678
        assert make_grid_id(20.12351, -15.56784) == "20.1235_-15.5678"

    def test_make_sample_id(self) -> None:
        sid = make_sample_id("Sahara", "20.0000_10.0000", 2020, 7)
        assert sid == "Sahara_20.0000_10.0000_2020_07"

    def test_make_sample_id_zero_pads_month(self) -> None:
        sid = make_sample_id("Amazon", "0.0_-60.0", 2019, 1)
        assert sid.endswith("_01")


# ---------------------------------------------------------------------------
# Core window builder tests
# ---------------------------------------------------------------------------


class TestBuildForecastingSamples:
    """Tests for the main build_forecasting_samples function."""

    def test_one_grid_cell_12_months_yields_6_samples(self) -> None:
        """12 months → windows: months 1-6→7, 2-7→8, …, 6-11→12 = 6 samples."""
        data = make_tiny_grid_cell_data(n_months=12)
        samples, meta = build_forecasting_samples(data, sequence_length=6)
        assert len(samples) == 6
        assert meta.total_samples == 6
        assert meta.grid_cells == 1

    def test_one_grid_cell_24_months_yields_18_samples(self) -> None:
        """24 months → 24 - 6 = 18 samples."""
        data = make_tiny_grid_cell_data(n_months=24)
        samples, meta = build_forecasting_samples(data, sequence_length=6)
        assert len(samples) == 18

    def test_too_few_months_dropped(self) -> None:
        """A grid cell with only 6 months cannot produce any sample."""
        data = make_tiny_grid_cell_data(n_months=6)
        with pytest.raises(ValueError, match="No forecasting samples"):
            build_forecasting_samples(data, sequence_length=6)

    def test_exactly_7_months_yields_1_sample(self) -> None:
        """7 months = 1 sample (months 1-6 → month 7)."""
        data = make_tiny_grid_cell_data(n_months=7)
        samples, meta = build_forecasting_samples(data, sequence_length=6)
        assert len(samples) == 1

    def test_y_true_equals_target_month_evaporation(self) -> None:
        """y_true must be the evaporation_anomaly at the target month."""
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        # First sample: months 1-6 → target month 7, value = 7 * 10 = 70
        assert samples.iloc[0]["y_true"] == pytest.approx(70.0)
        # Second sample: months 2-7 → target month 8, value = 80
        assert samples.iloc[1]["y_true"] == pytest.approx(80.0)
        # Last sample: months 6-11 → target month 12, value = 120
        assert samples.iloc[-1]["y_true"] == pytest.approx(120.0)

    def test_lag_1_is_previous_month(self) -> None:
        """lag_1 = t-1 (the month immediately before target)."""
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        # First sample: target = month 7 (value=7).
        # lag_1 should be month 6, value = 6.
        first = samples.iloc[0]
        target_month = first["target_month"]
        assert target_month == 7
        assert first["temperature_anomaly_lag_1"] == pytest.approx(6.0)
        # Second sample: target = month 8 (value=8).
        # lag_1 should be month 7, value = 7.
        second = samples.iloc[1]
        assert second["temperature_anomaly_lag_1"] == pytest.approx(7.0)

    def test_lag_6_is_six_months_before_target(self) -> None:
        """lag_6 = t-6 (six months before target month)."""
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        # First sample: target = month 7.
        # lag_6 = t-6 = month 1, value = 1.
        first = samples.iloc[0]
        assert first["temperature_anomaly_lag_6"] == pytest.approx(1.0)
        # Last sample: target = month 12.
        # lag_6 = t-6 = month 6, value = 6.
        last = samples.iloc[-1]
        assert last["temperature_anomaly_lag_6"] == pytest.approx(6.0)

    def test_all_lags_present_for_each_feature(self) -> None:
        """Every (feature, lag_1 … lag_6) column must exist."""
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        for feat in DEFAULT_FEATURE_COLUMNS:
            for lag in range(1, 7):
                assert f"{feat}_lag_{lag}" in samples.columns

    def test_sample_id_uniqueness(self) -> None:
        data = make_tiny_grid_cell_data(n_months=24)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        assert samples["sample_id"].is_unique
        assert len(samples["sample_id"]) == len(samples)

    def test_grid_id_stable(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        expected_grid_id = make_grid_id(20.0, 10.0)
        assert (samples["grid_id"] == expected_grid_id).all()

    def test_no_future_leakage(self) -> None:
        """input_window_end must be strictly before the target month."""
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        for _, row in samples.iterrows():
            input_end = str(row["input_window_end"])
            target = f"{int(row['target_year'])}-{int(row['target_month']):02d}"
            assert input_end < target, f"Leakage: {input_end} >= {target}"

    def test_input_window_covers_exactly_6_months(self) -> None:
        """Window start to window end should span exactly 5 months (6 data points)."""
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        first = samples.iloc[0]
        # input_window_start = "2020-01", input_window_end = "2020-06"
        assert first["input_window_start"] == "2020-01"
        assert first["input_window_end"] == "2020-06"

    def test_cross_year_boundary(self) -> None:
        """Samples that cross year boundaries should work correctly."""
        data = make_tiny_grid_cell_data(n_months=12, start_year=2019, start_month=10)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        assert len(samples) == 6
        # First sample: target = month 4 of year 2020
        first = samples.iloc[0]
        assert first["target_year"] == 2020
        assert first["target_month"] == 4
        # input_window_start = "2019-10"
        assert first["input_window_start"] == "2019-10"

    def test_multiple_grid_cells(self) -> None:
        """Two grid cells, each 12 months → 12 samples total."""
        rows1 = make_tiny_grid_cell_data(n_months=12, lat=20.0, lon=10.0)
        rows2 = make_tiny_grid_cell_data(n_months=12, lat=21.0, lon=11.0)
        data = pd.concat([rows1, rows2], ignore_index=True)
        samples, meta = build_forecasting_samples(data, sequence_length=6)
        assert len(samples) == 12
        assert meta.grid_cells == 2

    def test_static_columns_from_target_month(self) -> None:
        """month_sin and month_cos come from the target month."""
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        first = samples.iloc[0]
        # target month = 7 → month_sin = sin(2π*7/12)
        expected_sin = np.sin(2 * np.pi * 7 / 12)
        assert first["month_sin"] == pytest.approx(expected_sin)

    def test_climate_type_from_registry(self) -> None:
        """climate_type should come from the region registry."""
        data = make_tiny_grid_cell_data(n_months=12, region="Sahara")
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        assert (samples["climate_type"] == "arid").all()

    def test_climate_type_east_china(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12, region="East China")
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        assert (samples["climate_type"] == "monsoon").all()


# ---------------------------------------------------------------------------
# 3D sequence array tests
# ---------------------------------------------------------------------------


class TestBuildSequenceArrays:
    def test_output_shapes(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        X, y = build_sequence_arrays_from_samples(samples, sequence_length=6)
        assert X.shape == (6, 6, 7)  # 6 samples, 6 lags, 7 features
        assert y.shape == (6,)

    def test_sequence_order_preserved(self) -> None:
        """First sample: sequence[:, 0] = temperature_anomaly values 1..6."""
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        X, y = build_sequence_arrays_from_samples(samples, sequence_length=6)
        # temperature_anomaly is feature index 0
        # X[0, :, 0] = [1, 2, 3, 4, 5, 6] (lag_6 → lag_1 order)
        # Wait — let's check: in build_sequence_arrays_from_samples,
        # lag_idx=1 fills array_idx=0, lag_idx=2 fills array_idx=1, ...
        # So X[0, 0, :] = lag_1 values, X[0, 5, :] = lag_6 values.
        # First sample target = month 7, so:
        #   array_idx 0 (lag_1) = month 6, temp = 6
        #   array_idx 5 (lag_6) = month 1, temp = 1
        assert X[0, 0, 0] == pytest.approx(6.0)  # lag_1 temp
        assert X[0, 5, 0] == pytest.approx(1.0)  # lag_6 temp


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateSamples:
    def test_valid_samples_pass(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        errors = validate_samples(samples, sequence_length=6)
        assert errors == []

    def test_duplicate_ids_detected(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        # Duplicate first row
        dup = pd.concat([samples, samples.iloc[[0]]], ignore_index=True)
        errors = validate_samples(dup, sequence_length=6)
        assert any("Duplicate sample_id" in e for e in errors)

    def test_missing_lag_detected(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        samples.drop(columns=["temperature_anomaly_lag_3"], inplace=True)
        errors = validate_samples(samples, sequence_length=6)
        assert any("Missing lag column" in e for e in errors)


# ---------------------------------------------------------------------------
# I/O tests
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12)
        samples, meta = build_forecasting_samples(data, sequence_length=6)

        with tempfile.TemporaryDirectory() as tmp:
            paths = save_forecasting_dataset(samples, meta, tmp, prefix="test")
            assert paths["samples"].exists()
            assert paths["metadata"].exists()

            loaded_df, loaded_meta = load_forecasting_dataset(
                paths["samples"], paths["metadata"]
            )
            assert len(loaded_df) == len(samples)
            assert loaded_meta is not None
            assert loaded_meta["total_samples"] == meta.total_samples

    def test_build_and_save(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12)
        with tempfile.TemporaryDirectory() as tmp:
            samples, meta, errors = build_and_save(
                data, output_dir=tmp, prefix="test"
            )
            assert len(samples) == 6
            assert errors == []
            assert (Path(tmp) / "test_samples.csv").exists()
            assert (Path(tmp) / "test_metadata.json").exists()


# ---------------------------------------------------------------------------
# Cross-year and edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_grid_cell_returns_correct_columns(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12)
        samples, _ = build_forecasting_samples(data, sequence_length=6)
        expected_cols = {
            "sample_id",
            "grid_id",
            "region",
            "target_year",
            "target_month",
            "latitude",
            "longitude",
            "climate_type",
            "input_window_start",
            "input_window_end",
            "month_sin",
            "month_cos",
            "y_true",
        }
        for feat in DEFAULT_FEATURE_COLUMNS:
            for lag in range(1, 7):
                expected_cols.add(f"{feat}_lag_{lag}")
        actual = set(samples.columns)
        missing = expected_cols - actual
        assert not missing, f"Missing columns: {missing}"

    def test_missing_required_column_raises(self) -> None:
        data = make_tiny_grid_cell_data(n_months=12)
        data.drop(columns=["temperature_anomaly"], inplace=True)
        with pytest.raises(ValueError, match="Missing required columns"):
            build_forecasting_samples(data, sequence_length=6)

    def test_empty_dataframe_raises(self) -> None:
        data = pd.DataFrame(columns=["region", "year", "month", "latitude", "longitude"])
        with pytest.raises(ValueError):
            build_forecasting_samples(data, sequence_length=6)
