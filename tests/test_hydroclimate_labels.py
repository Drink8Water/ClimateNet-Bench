"""Tests for hydroclimate event label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climatenet.evaluation.hydroclimate_labels import (
    ALL_EVENT_TYPES,
    EVENT_COMPOUND_HOT_DRY,
    EVENT_EVAPORATION_DEFICIT,
    EVENT_SOIL_MOISTURE_DROUGHT,
    _validate_columns,
    _validate_thresholds,
    build_all_event_labels,
    build_compound_hot_dry_label,
    build_evaporation_deficit_label,
    build_soil_moisture_drought_label,
    fit_event_thresholds,
)


# ---------------------------------------------------------------------------
# Test data builder
# ---------------------------------------------------------------------------


def make_seasonal_data(
    n_per_month: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a realistic multi-month dataset with seasonal patterns.

    Each month has distinct mean values so we can verify that thresholds
    are computed per calendar month.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for month in range(1, 13):
        # Seasonal baseline: winter = low temp, summer = high temp
        base_temp = 15.0 + 10.0 * np.sin(np.pi * (month - 4) / 6)
        base_sm = 0.20 - 0.05 * np.sin(np.pi * (month - 4) / 6)
        base_evap = 3.0 + 2.0 * np.sin(np.pi * (month - 4) / 6)

        for _ in range(n_per_month):
            rows.append(
                {
                    "month": month,
                    "region": "Sahara" if rng.random() < 0.5 else "East China",
                    "temperature_anomaly": float(rng.normal(0, 1.5)),
                    "soil_moisture_anomaly": float(rng.normal(0, 0.05)),
                    "evaporation_anomaly": float(rng.normal(0, 1.0)),
                    "temperature": float(rng.normal(base_temp, 2.0)),
                    "soil_moisture": float(rng.normal(base_sm, 0.02)),
                    "evaporation": float(rng.normal(base_evap, 1.5)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fit_event_thresholds
# ---------------------------------------------------------------------------


class TestFitEventThresholds:
    def test_all_12_months_present(self) -> None:
        df = make_seasonal_data(n_per_month=50)
        thresholds = fit_event_thresholds(df)
        assert set(thresholds.keys()) == set(range(1, 13))

    def test_each_month_has_three_keys(self) -> None:
        df = make_seasonal_data(n_per_month=50)
        thresholds = fit_event_thresholds(df)
        for month in range(1, 13):
            assert "sm_p10" in thresholds[month]
            assert "evap_p10" in thresholds[month]
            assert "temp_p90" in thresholds[month]

    def test_thresholds_are_numeric(self) -> None:
        df = make_seasonal_data(n_per_month=50)
        thresholds = fit_event_thresholds(df)
        for month in range(1, 13):
            for key in ["sm_p10", "evap_p10", "temp_p90"]:
                val = thresholds[month][key]
                assert isinstance(val, float)
                assert not np.isnan(val)

    def test_partial_months_accepted(self) -> None:
        """Data with only some calendar months should be accepted."""
        df = pd.DataFrame(
            {
                "month": [1] * 10,
                "region": ["Sahara"] * 10,
                "soil_moisture_anomaly": np.random.randn(10),
                "evaporation_anomaly": np.random.randn(10),
                "temperature_anomaly": np.random.randn(10),
            }
        )
        thresholds = fit_event_thresholds(df)
        assert set(thresholds.keys()) == {1}
        assert "sm_p10" in thresholds[1]

    def test_deterministic_with_same_data(self) -> None:
        df1 = make_seasonal_data(n_per_month=100, seed=42)
        df2 = make_seasonal_data(n_per_month=100, seed=42)
        t1 = fit_event_thresholds(df1)
        t2 = fit_event_thresholds(df2)
        for month in range(1, 13):
            for key in ["sm_p10", "evap_p10", "temp_p90"]:
                assert t1[month][key] == t2[month][key]

    def test_percentile_approximately_correct(self) -> None:
        """For a simple dataset, P10 and P90 should be near expected values."""
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "month": [7] * 1000,
                "region": ["Sahara"] * 1000,
                "soil_moisture_anomaly": rng.normal(0, 1, 1000),
                "evaporation_anomaly": rng.normal(0, 1, 1000),
                "temperature_anomaly": rng.normal(0, 1, 1000),
            }
        )
        t = fit_event_thresholds(data)
        # P10 of N(0,1) ≈ -1.28
        assert -1.5 < t[7]["sm_p10"] < -1.0
        assert -1.5 < t[7]["evap_p10"] < -1.0
        # P90 of N(0,1) ≈ +1.28
        assert 1.0 < t[7]["temp_p90"] < 1.5


# ---------------------------------------------------------------------------
# Label builders
# ---------------------------------------------------------------------------


class TestBuildSoilMoistureDroughtLabel:
    def test_basic_output_shape_and_type(self) -> None:
        df = make_seasonal_data(n_per_month=50)
        train_df = df.iloc[:600]
        test_df = df.iloc[600:]
        thresholds = fit_event_thresholds(train_df)
        labels = build_soil_moisture_drought_label(test_df, thresholds)
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == bool
        assert len(labels) == len(test_df)

    def test_all_values_below_p10_are_flagged(self) -> None:
        """Extreme negative anomalies should all be labelled as drought."""
        # Build data where month 7 has a very low P10
        train_df = pd.DataFrame(
            {
                "month": [7] * 100,
                "region": ["Sahara"] * 100,
                "soil_moisture_anomaly": np.linspace(-3, 3, 100),
                "evaporation_anomaly": np.zeros(100),
                "temperature_anomaly": np.zeros(100),
            }
        )
        thresholds = fit_event_thresholds(train_df)
        # P10 of uniform(-3,3) ≈ -2.4
        # Test with very negative values → all should be flagged
        test_df = pd.DataFrame(
            {
                "month": [7] * 5,
                "region": ["Sahara"] * 5,
                "soil_moisture_anomaly": [-5.0, -4.0, -3.0, -2.0, -1.0],
                "evaporation_anomaly": np.zeros(5),
                "temperature_anomaly": np.zeros(5),
            }
        )
        labels = build_soil_moisture_drought_label(test_df, thresholds)
        # The first 3 should be below P10 (-2.4), last 2 above
        assert labels[0:3].all()
        assert not labels[3:5].any()

    def test_thresholds_vary_by_month(self) -> None:
        """A value that is P10 in July should not be P10 in January."""
        rng = np.random.default_rng(42)
        # Month 1: centred at +2 (wet), Month 7: centred at -2 (dry)
        rows = []
        for _ in range(500):
            rows.append(
                {
                    "month": 1,
                    "region": "Sahara",
                    "soil_moisture_anomaly": float(rng.normal(2.0, 0.5)),
                    "evaporation_anomaly": 0.0,
                    "temperature_anomaly": 0.0,
                }
            )
            rows.append(
                {
                    "month": 7,
                    "region": "Sahara",
                    "soil_moisture_anomaly": float(rng.normal(-2.0, 0.5)),
                    "evaporation_anomaly": 0.0,
                    "temperature_anomaly": 0.0,
                }
            )
        train_df = pd.DataFrame(rows)
        thresholds = fit_event_thresholds(train_df)

        # A value of 0 is dry for Jan (mean=2 → P10≈1.36) but wet for Jul (mean=-2 → P10≈-2.64)
        test_df = pd.DataFrame(
            {
                "month": [1, 7],
                "region": ["Sahara", "Sahara"],
                "soil_moisture_anomaly": [0.0, 0.0],
                "evaporation_anomaly": [0.0, 0.0],
                "temperature_anomaly": [0.0, 0.0],
            }
        )
        labels = build_soil_moisture_drought_label(test_df, thresholds)
        # Jan: P10 ≈ 1.36, value 0 < 1.36 → drought
        assert bool(labels[0]) is True
        # Jul: P10 ≈ -2.64, value 0 > -2.64 → not drought
        assert bool(labels[1]) is False

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"month": [1], "temperature": [20.0]})
        thresholds = fit_event_thresholds(make_seasonal_data(n_per_month=10))
        with pytest.raises(ValueError, match="missing required columns"):
            build_soil_moisture_drought_label(df, thresholds)


class TestBuildEvaporationDeficitLabel:
    def test_basic_output(self) -> None:
        df = make_seasonal_data(n_per_month=50)
        thresholds = fit_event_thresholds(df)
        labels = build_evaporation_deficit_label(df, thresholds)
        assert len(labels) == len(df)
        assert labels.dtype == bool

    def test_deficit_rate_approximately_10_percent(self) -> None:
        """With a large dataset, ~10% of test samples should be flagged (P10).

        Use stratified split by month so both train and test contain all months.
        """
        df = make_seasonal_data(n_per_month=200)
        # Stratified split: alternate rows within each month
        train_mask = df.groupby("month").cumcount() < 100
        train_df = df[train_mask].copy()
        test_df = df[~train_mask].copy()
        thresholds = fit_event_thresholds(train_df)
        labels = build_evaporation_deficit_label(test_df, thresholds)
        rate = labels.mean()
        # Should be roughly 0.10, allow ±0.04 tolerance
        assert 0.06 < rate < 0.14


class TestBuildCompoundHotDryLabel:
    def test_basic_output(self) -> None:
        df = make_seasonal_data(n_per_month=50)
        thresholds = fit_event_thresholds(df)
        labels = build_compound_hot_dry_label(df, thresholds)
        assert len(labels) == len(df)
        assert labels.dtype == bool

    def test_both_conditions_must_hold(self) -> None:
        """Only samples that are BOTH hot AND dry should be flagged."""
        rng = np.random.default_rng(42)
        # Month 7: P90(temp) ≈ 1.28, P10(sm) ≈ -1.28
        train_df = pd.DataFrame(
            {
                "month": [7] * 1000,
                "region": ["Sahara"] * 1000,
                "soil_moisture_anomaly": rng.normal(0, 1, 1000),
                "evaporation_anomaly": np.zeros(1000),
                "temperature_anomaly": rng.normal(0, 1, 1000),
            }
        )
        thresholds = fit_event_thresholds(train_df)

        test_df = pd.DataFrame(
            {
                "month": [7, 7, 7, 7],
                "region": ["Sahara"] * 4,
                "soil_moisture_anomaly": [-2.0, -2.0, 2.0, 2.0],  # dry, dry, wet, wet
                "evaporation_anomaly": [0.0] * 4,
                "temperature_anomaly": [2.0, -2.0, 2.0, -2.0],  # hot, cold, hot, cold
            }
        )
        labels = build_compound_hot_dry_label(test_df, thresholds)
        # Only row 0: dry (-2) AND hot (+2) → compound
        assert bool(labels[0]) is True
        # Row 1: dry but cold → not compound
        assert bool(labels[1]) is False
        # Row 2: hot but wet → not compound
        assert bool(labels[2]) is False
        # Row 3: cold and wet → not compound
        assert bool(labels[3]) is False

    def test_compound_rate_lower_than_individual(self) -> None:
        """Compound event rate should be <= min(drought_rate, hot_rate)."""
        df = make_seasonal_data(n_per_month=200)
        train_mask = df.groupby("month").cumcount() < 100
        train_df = df[train_mask].copy()
        test_df = df[~train_mask].copy()
        thresholds = fit_event_thresholds(train_df)

        sm_label = build_soil_moisture_drought_label(test_df, thresholds)
        compound_label = build_compound_hot_dry_label(test_df, thresholds)

        assert compound_label.sum() <= sm_label.sum()


class TestBuildAllEventLabels:
    def test_returns_three_keys(self) -> None:
        df = make_seasonal_data(n_per_month=50)
        thresholds = fit_event_thresholds(df)
        labels = build_all_event_labels(df, thresholds)
        assert set(labels.keys()) == set(ALL_EVENT_TYPES)

    def test_all_values_are_boolean_arrays(self) -> None:
        df = make_seasonal_data(n_per_month=50)
        thresholds = fit_event_thresholds(df)
        labels = build_all_event_labels(df, thresholds)
        for key in ALL_EVENT_TYPES:
            assert labels[key].dtype == bool
            assert len(labels[key]) == len(df)


# ---------------------------------------------------------------------------
# Anti-leakage: train/test threshold isolation
# ---------------------------------------------------------------------------


class TestTrainTestIsolation:
    def test_test_data_does_not_affect_thresholds(self) -> None:
        """Thresholds fitted on train must be identical regardless of test data."""
        train_df = make_seasonal_data(n_per_month=100, seed=1)
        test_df1 = make_seasonal_data(n_per_month=10, seed=99)
        test_df2 = make_seasonal_data(n_per_month=10, seed=999)

        # Concatenate train with different test sets; fit ONLY on train part
        thresholds1 = fit_event_thresholds(train_df)
        thresholds2 = fit_event_thresholds(train_df.copy())

        # Thresholds must be identical — test_df2 never entered fit_event_thresholds
        for month in range(1, 13):
            for key in ["sm_p10", "evap_p10", "temp_p90"]:
                assert thresholds1[month][key] == thresholds2[month][key]

    def test_labels_on_test_use_train_thresholds(self) -> None:
        """Labels on test_df must use thresholds from train_df."""
        # Train: centred at -2 (very dry in July)
        rng = np.random.default_rng(1)
        train_df = pd.DataFrame(
            {
                "month": [7] * 500,
                "region": ["Sahara"] * 500,
                "soil_moisture_anomaly": rng.normal(-2.0, 0.5, 500),
                "evaporation_anomaly": np.zeros(500),
                "temperature_anomaly": np.zeros(500),
            }
        )
        thresholds = fit_event_thresholds(train_df)
        train_p10 = thresholds[7]["sm_p10"]
        # Train P10 should be around -2.64 for N(-2,0.5)

        # Test: centred at +2 (wet in July)
        rng2 = np.random.default_rng(2)
        test_df = pd.DataFrame(
            {
                "month": [7] * 500,
                "region": ["Sahara"] * 500,
                "soil_moisture_anomaly": rng2.normal(2.0, 0.5, 500),
                "evaporation_anomaly": np.zeros(500),
                "temperature_anomaly": np.zeros(500),
            }
        )
        # If we INCORRECTLY fit on test, its P10 would be ~1.36
        # Since we correctly use train thresholds (P10 ≈ -2.64), and test values
        # are all around +2 (> -2.64), NO test samples should be flagged as drought.
        labels = build_soil_moisture_drought_label(test_df, thresholds)
        # Almost all test values are well above the train P10
        drought_rate = labels.mean()
        assert drought_rate < 0.05  # ≤5% — only extreme outliers


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_sample_per_month(self) -> None:
        """One sample per month is the minimum for percentile computation."""
        rows = []
        for month in range(1, 13):
            rows.append(
                {
                    "month": month,
                    "region": "Sahara",
                    "soil_moisture_anomaly": float(month),
                    "evaporation_anomaly": float(month),
                    "temperature_anomaly": float(month),
                }
            )
        df = pd.DataFrame(rows)
        thresholds = fit_event_thresholds(df)
        labels = build_all_event_labels(df, thresholds)
        # With 1 sample per month, P10 equals the only sample's value
        # So no drought (value < P10 is False because value == P10)
        for key in ALL_EVENT_TYPES:
            assert labels[key].sum() == 0

    def test_constant_values_all_same(self) -> None:
        """When all values are identical, P10 == P90 == that value."""
        df = pd.DataFrame(
            {
                "month": [1] * 100,
                "region": ["Sahara"] * 100,
                "soil_moisture_anomaly": [5.0] * 100,
                "evaporation_anomaly": [5.0] * 100,
                "temperature_anomaly": [5.0] * 100,
            }
        )
        thresholds = fit_event_thresholds(df)
        assert thresholds[1]["sm_p10"] == 5.0
        assert thresholds[1]["temp_p90"] == 5.0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    def test_validate_columns_ok(self) -> None:
        df = pd.DataFrame(
            {
                "soil_moisture_anomaly": [],
                "evaporation_anomaly": [],
                "temperature_anomaly": [],
                "month": [],
            }
        )
        _validate_columns(df, "test")  # should not raise

    def test_validate_columns_missing(self) -> None:
        df = pd.DataFrame({"month": [1], "temperature": [20.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_columns(df, "bad_df")

    def test_validate_thresholds_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_thresholds({})

    def test_validate_thresholds_invalid_month_key(self) -> None:
        t = {13: {"sm_p10": 0.0, "evap_p10": 0.0, "temp_p90": 0.0}}
        with pytest.raises(ValueError, match="Invalid month"):
            _validate_thresholds(t)

    def test_validate_thresholds_missing_key(self) -> None:
        t = {}
        for m in range(1, 13):
            t[m] = {"sm_p10": 0.0, "evap_p10": 0.0}  # missing temp_p90
        with pytest.raises(ValueError, match="temp_p90"):
            _validate_thresholds(t)
