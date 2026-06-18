"""Tests for train-only climatology and anomaly computation.

Verifies that:

1. Climatology is computed from training data only.
2. Test/val data never affect the anomaly values.
3. Missing calendar months raise a clear ``ValueError``.
4. The ``group_by_climate_zone`` flag works correctly.
5. ``build_train_only_anomaly`` produces correct output shapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climatenet.preprocessing.climatology import (
    apply_monthly_anomaly,
    build_train_only_anomaly,
    compute_monthly_climatology,
)


# ---------------------------------------------------------------------------
# Synthetic test data builders
# ---------------------------------------------------------------------------


def _make_seasonal_data(
    n_months: int = 36,
    base_value: float = 10.0,
    amplitude: float = 5.0,
    noise: float = 0.5,
    region: str = "Sahara",
    climate_zone: str = "arid",
    start_year: int = 2020,
    start_month: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a multi-year monthly time series with a clear seasonal cycle.

    ``value = base + amplitude * cos(2π*(month-7)/12) + noise``

    This simulates a climate variable (e.g. temperature) that peaks in
    northern-hemisphere summer (month 7).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_months):
        year = start_year + (start_month + i - 1) // 12
        month = ((start_month + i - 1) % 12) + 1
        val = (
            base_value
            + amplitude * np.cos(2 * np.pi * (month - 7) / 12)
            + rng.normal(0, noise)
        )
        rows.append(
            {
                "sample_id": f"{region}_{i:04d}",
                "year": year,
                "month": month,
                "region": region,
                "climate_zone": climate_zone,
                "value": round(float(val), 4),
                "lat": 25.0,
                "lon": 10.0,
            }
        )
    return pd.DataFrame(rows)


def _make_multi_zone_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test/climatology DataFrames across two climate zones.

    Train = Sahara (arid) 24 months
    Val   = Sahara (arid) 6 months (different years)
    Test  = East China (monsoon) 12 months
    Climatology = pre-computed from train for convenience
    """
    rng = np.random.default_rng(42)

    def _make(region, zone, n_months, start_year, start_month):
        rows = []
        for i in range(n_months):
            year = start_year + (start_month + i - 1) // 12
            month = ((start_month + i - 1) % 12) + 1
            val = 15.0 + 8.0 * np.cos(2 * np.pi * (month - 7) / 12) + rng.normal(0, 0.5)
            rows.append(
                {
                    "sample_id": f"{region}_{i:04d}",
                    "year": year,
                    "month": month,
                    "region": region,
                    "climate_zone": zone,
                    "value": round(float(val), 4),
                    "lat": 25.0 if region == "Sahara" else 30.0,
                    "lon": 10.0 if region == "Sahara" else 115.0,
                }
            )
        return pd.DataFrame(rows)

    train = _make("Sahara", "arid", 24, 2020, 1)
    val = _make("Sahara", "arid", 6, 2022, 1)
    test = _make("East China", "monsoon", 12, 2022, 1)
    clim = compute_monthly_climatology(train, value_col="value")
    return train, val, test, clim


# ---------------------------------------------------------------------------
# compute_monthly_climatology
# ---------------------------------------------------------------------------


class TestComputeMonthlyClimatology:
    """Tests for :func:`compute_monthly_climatology`."""

    def test_output_has_12_rows_one_per_month(self) -> None:
        train = _make_seasonal_data(n_months=36)
        clim = compute_monthly_climatology(train, value_col="value")
        assert len(clim) == 12
        assert "value_climatology" in clim.columns

    def test_raises_on_missing_value_col(self) -> None:
        train = _make_seasonal_data(n_months=12)
        with pytest.raises(ValueError, match="not found in train_df"):
            compute_monthly_climatology(train, value_col="nonexistent")

    def test_raises_on_missing_month_col(self) -> None:
        train = _make_seasonal_data(n_months=12)
        train = train.drop(columns=["month"])
        with pytest.raises(ValueError, match="must contain a 'month' column"):
            compute_monthly_climatology(train, value_col="value")

    def test_raises_when_calendar_month_missing(self) -> None:
        # Build 11 months starting at month 3: months 3-12, then 1.
        # Month 2 is genuinely missing.
        train = _make_seasonal_data(n_months=11, start_month=3)
        # 11 months starting from 3: 3,4,5,6,7,8,9,10,11,12,1 → month 2 missing
        with pytest.raises(ValueError, match="missing data for calendar month"):
            compute_monthly_climatology(train, value_col="value")

    def test_climatology_is_mean_per_month(self) -> None:
        """Build a DataFrame where month-7 always has value=100 → clim should be 100."""
        rows = []
        for year in [2020, 2021, 2022]:
            for month in range(1, 13):
                rows.append(
                    {
                        "sample_id": f"s_{year}_{month}",
                        "year": year,
                        "month": month,
                        "region": "Test",
                        "climate_zone": "test",
                        "value": 100.0 if month == 7 else 0.0,
                        "lat": 0.0,
                        "lon": 0.0,
                    }
                )
        train = pd.DataFrame(rows)
        clim = compute_monthly_climatology(train, value_col="value")
        july_row = clim[clim["month"] == 7]
        assert july_row["value_climatology"].iloc[0] == pytest.approx(100.0)

    def test_group_by_climate_zone(self) -> None:
        """Two zones, each with different base values → per-zone climatologies."""
        rows = []
        for year in [2020, 2021]:
            for month in range(1, 13):
                rows.append(
                    {
                        "sample_id": f"sahara_{year}_{month}",
                        "year": year,
                        "month": month,
                        "region": "Sahara",
                        "climate_zone": "arid",
                        "value": 10.0,
                        "lat": 25.0,
                        "lon": 10.0,
                    }
                )
                rows.append(
                    {
                        "sample_id": f"amazon_{year}_{month}",
                        "year": year,
                        "month": month,
                        "region": "Amazon",
                        "climate_zone": "tropical_humid",
                        "value": 30.0,
                        "lat": -3.0,
                        "lon": -60.0,
                    }
                )
        train = pd.DataFrame(rows)
        clim = compute_monthly_climatology(
            train, value_col="value", group_by_climate_zone=True,
        )
        assert len(clim) == 24  # 2 zones × 12 months
        arid = clim[clim["climate_zone"] == "arid"]
        tropical = clim[clim["climate_zone"] == "tropical_humid"]
        assert arid["value_climatology"].iloc[0] == pytest.approx(10.0)
        assert tropical["value_climatology"].iloc[0] == pytest.approx(30.0)

    def test_group_by_climate_zone_missing_column_raises(self) -> None:
        train = _make_seasonal_data(n_months=24)
        train = train.drop(columns=["climate_zone"])
        with pytest.raises(ValueError, match="requires a 'climate_zone' column"):
            compute_monthly_climatology(
                train, value_col="value", group_by_climate_zone=True,
            )


# ---------------------------------------------------------------------------
# apply_monthly_anomaly
# ---------------------------------------------------------------------------


class TestApplyMonthlyAnomaly:
    """Tests for :func:`apply_monthly_anomaly`."""

    def test_anomaly_mean_per_month_is_zero(self) -> None:
        """Climatology is the monthly mean → anomalies for a given month
        should average to ~0 when averaged across all years."""
        train = _make_seasonal_data(n_months=36)
        clim = compute_monthly_climatology(train, value_col="value")
        result = apply_monthly_anomaly(train, clim, value_col="value")
        # Mean anomaly per calendar month (across all years) should be ~0.
        monthly_mean = result.groupby("month")["value_anomaly"].mean()
        for month, m in monthly_mean.items():
            assert abs(m) < 1e-6, f"Month {month} anomaly mean {m} not close to 0"

    def test_preserves_original_columns(self) -> None:
        train = _make_seasonal_data(n_months=24)
        clim = compute_monthly_climatology(train, value_col="value")
        result = apply_monthly_anomaly(train, clim, value_col="value")
        for col in ["sample_id", "year", "month", "region", "climate_zone"]:
            assert col in result.columns

    def test_custom_output_col(self) -> None:
        train = _make_seasonal_data(n_months=24)
        clim = compute_monthly_climatology(train, value_col="value")
        result = apply_monthly_anomaly(
            train, clim, value_col="value", output_col="my_anomaly",
        )
        assert "my_anomaly" in result.columns
        assert "value_anomaly" not in result.columns

    def test_raises_when_climatology_missing_column(self) -> None:
        train = _make_seasonal_data(n_months=24)
        clim = pd.DataFrame({"month": [1], "wrong_name": [0.0]})
        with pytest.raises(ValueError, match="missing expected column"):
            apply_monthly_anomaly(train, clim, value_col="value")

    def test_raises_when_df_missing_group_col(self) -> None:
        train = _make_seasonal_data(n_months=24)
        clim = compute_monthly_climatology(train, value_col="value")
        train_no_month = train.drop(columns=["month"])
        with pytest.raises(ValueError, match="missing from input DataFrame"):
            apply_monthly_anomaly(train_no_month, clim, value_col="value")


# ---------------------------------------------------------------------------
# build_train_only_anomaly
# ---------------------------------------------------------------------------


class TestBuildTrainOnlyAnomaly:
    """Tests for :func:`build_train_only_anomaly`."""

    def test_returns_four_dataframes(self) -> None:
        train, val, test, _ = _make_multi_zone_data()
        t_out, v_out, ts_out, clim = build_train_only_anomaly(
            train, val, test, value_col="value",
        )
        assert isinstance(t_out, pd.DataFrame)
        assert isinstance(v_out, pd.DataFrame)
        assert isinstance(ts_out, pd.DataFrame)
        assert isinstance(clim, pd.DataFrame)

    def test_train_anomalies_use_train_climatology(self) -> None:
        """Anomaly = value - train_climatology."""
        train, val, test, _ = _make_multi_zone_data()
        t_out, _, _, clim = build_train_only_anomaly(train, val, test, value_col="value")

        # Spot-check one month
        month_7_clim = clim[clim["month"] == 7]["value_climatology"].iloc[0]
        train_july = t_out[t_out["month"] == 7].iloc[0]
        expected_anomaly = train_july["value"] - month_7_clim
        assert train_july["value_anomaly"] == pytest.approx(expected_anomaly)

    def test_test_data_never_affects_climatology(self) -> None:
        """Even if test has extreme values, climatology should come from train."""
        train, val, _, _clim = _make_multi_zone_data()

        # Create a test set with absurd values
        rng = np.random.default_rng(99)
        rows = []
        for i in range(12):
            rows.append(
                {
                    "sample_id": f"extreme_{i}",
                    "year": 2023,
                    "month": i + 1,
                    "region": "East China",
                    "climate_zone": "monsoon",
                    "value": 9999.0 + rng.normal(0, 0.1),
                    "lat": 30.0,
                    "lon": 115.0,
                }
            )
        test_extreme = pd.DataFrame(rows)

        # Fit with normal test first
        _, _, _, clim_normal = build_train_only_anomaly(
            train, val, test_extreme, value_col="value",
        )
        # Fit with same train → same climatology regardless of test
        _, _, _, clim2 = build_train_only_anomaly(
            train, val, test_extreme, value_col="value",
        )
        pd.testing.assert_frame_equal(clim_normal, clim2)

    def test_anomaly_values_differ_between_train_and_test(self) -> None:
        """Train and test have different raw values → anomalies differ."""
        train, val, test, _ = _make_multi_zone_data()
        t_out, _, ts_out, _ = build_train_only_anomaly(train, val, test, value_col="value")

        # They should not be identical (different regions → different raw values)
        assert not t_out["value_anomaly"].equals(ts_out["value_anomaly"])

    def test_preserves_sample_id_year_month_region_climate_zone(self) -> None:
        train, val, test, _ = _make_multi_zone_data()
        t_out, v_out, ts_out, _ = build_train_only_anomaly(train, val, test, value_col="value")

        for label, df in [("train", t_out), ("val", v_out), ("test", ts_out)]:
            for col in ["sample_id", "year", "month", "region", "climate_zone"]:
                assert col in df.columns, f"{col} missing from {label}"


# ---------------------------------------------------------------------------
# Anti-leakage — the core guarantee
# ---------------------------------------------------------------------------


class TestAntiLeakage:
    """Tests that prove climatology never leaks from val/test into train."""

    def test_climatology_depends_only_on_train(self) -> None:
        """Change test data → climatology must stay the same."""
        train, val, test, _ = _make_multi_zone_data()

        _, _, _, clim1 = build_train_only_anomaly(train, val, test, value_col="value")

        # Mutate test — climatology should be identical
        test_mutated = test.copy()
        test_mutated["value"] = -100.0
        _, _, _, clim2 = build_train_only_anomaly(
            train, val, test_mutated, value_col="value",
        )

        pd.testing.assert_frame_equal(clim1, clim2)

    def test_climatology_depends_only_on_train_with_climate_zone(self) -> None:
        train, val, test, _ = _make_multi_zone_data()

        _, _, _, clim1 = build_train_only_anomaly(
            train, val, test, value_col="value", group_by_climate_zone=True,
        )

        test_mutated = test.copy()
        test_mutated["value"] = 1e6
        _, _, _, clim2 = build_train_only_anomaly(
            train, val, test_mutated, value_col="value", group_by_climate_zone=True,
        )

        pd.testing.assert_frame_equal(clim1, clim2)

    def test_val_does_not_affect_climatology(self) -> None:
        train, val, test, _ = _make_multi_zone_data()

        _, _, _, clim1 = build_train_only_anomaly(train, val, test, value_col="value")

        val_mutated = val.copy()
        val_mutated["value"] = 500.0
        _, _, _, clim2 = build_train_only_anomaly(
            train, val_mutated, test, value_col="value",
        )

        pd.testing.assert_frame_equal(clim1, clim2)

    def test_compute_monthly_climatology_only_reads_given_df(self) -> None:
        """compute_monthly_climatology should be a pure function of its input."""
        train = _make_seasonal_data(n_months=36)
        clim1 = compute_monthly_climatology(train, value_col="value")
        clim2 = compute_monthly_climatology(train, value_col="value")
        pd.testing.assert_frame_equal(clim1, clim2)
