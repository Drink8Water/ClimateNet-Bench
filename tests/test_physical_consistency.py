"""Tests for physical consistency audit."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.evaluation.physical_consistency import (
    AUDIT_FEATURES,
    check_monotonic_trend,
    compute_feature_sensitivity,
    compute_regional_sensitivity,
    is_consistent_with_expectation,
    run_physical_audit,
)


# ---------------------------------------------------------------------------
# Simple mock model
# ---------------------------------------------------------------------------


class MockPhysicallyPlausibleModel:
    """A mock model whose predictions respond plausibly to inputs.

    Uses coefficients that match physical expectations when columns
    are passed in a known order.  Also accepts a column-name→coefficient
    map for robustness.
    """

    # Column-name → coefficient mapping
    _coefs: dict[str, float] = {
        "temperature_anomaly_lag_1": 0.15,
        "precipitation_anomaly_lag_1": 0.10,
        "radiation_anomaly_lag_1": 0.20,
        "soil_moisture_anomaly_lag_1": 0.25,
        "wind_speed_lag_1": 0.05,
        "dryness_proxy_lag_1": -0.15,
        "saturation_vapor_pressure_lag_1": 0.05,
    }

    def __init__(self, noise: float = 0.02, column_names: list[str] | None = None, seed: int = 42):
        self.noise = noise
        self.column_names = column_names  # passed by test to know column order
        self.rng = np.random.default_rng(seed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        pred = np.zeros(n)
        if self.column_names is not None:
            for i, col in enumerate(self.column_names):
                if col in self._coefs:
                    pred += self._coefs[col] * X[:, i]
        else:
            # Fallback: assume columns in default order
            for i, (col, coef) in enumerate(self._coefs.items()):
                if i < X.shape[1]:
                    pred += coef * X[:, i]
        return pred + self.rng.normal(0, self.noise, size=n)


class MockBrokenModel:
    """A model that returns constant predictions — should fail the audit."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


# ---------------------------------------------------------------------------
# Test data builder
# ---------------------------------------------------------------------------


def make_audit_test_data(
    n_samples: int = 300, seed: int = 42
) -> tuple[pd.DataFrame, list[str]]:
    """Build a tiny DataFrame with audit-compatible columns."""
    rng = np.random.default_rng(seed)
    feature_cols = [
        "temperature_anomaly_lag_1",
        "precipitation_anomaly_lag_1",
        "radiation_anomaly_lag_1",
        "soil_moisture_anomaly_lag_1",
        "wind_speed_lag_1",
        "dryness_proxy_lag_1",
        "saturation_vapor_pressure_lag_1",
    ]
    regions = ["Sahara", "Amazon", "East China"]
    rows = []
    for i in range(n_samples):
        row = {"region": regions[i % 3]}
        for col in feature_cols:
            row[col] = rng.normal(0, 1)
        # Add lag_2..lag_6 variants for realism
        for feat_base in [
            "temperature_anomaly",
            "precipitation_anomaly",
            "radiation_anomaly",
            "soil_moisture_anomaly",
            "wind_speed",
            "dryness_proxy",
            "saturation_vapor_pressure",
        ]:
            for lag in range(2, 7):
                row[f"{feat_base}_lag_{lag}"] = rng.normal(0, 0.8)
        rows.append(row)
    return pd.DataFrame(rows), feature_cols


# ---------------------------------------------------------------------------
# Feature sensitivity
# ---------------------------------------------------------------------------


class TestFeatureSensitivity:
    def test_output_shapes(self) -> None:
        df, feature_cols = make_audit_test_data()
        model = MockPhysicallyPlausibleModel(column_names=feature_cols)
        curve = compute_feature_sensitivity(
            model, df, "radiation_anomaly_lag_1",
            feature_cols=feature_cols, n_points=10,
        )
        assert len(curve["feature_values"]) == 10
        assert len(curve["mean_prediction"]) == 10
        assert len(curve["std_prediction"]) == 10
        assert curve["feature"] == "radiation_anomaly_lag_1"

    def test_n_points_parameter(self) -> None:
        df, feature_cols = make_audit_test_data()
        model = MockPhysicallyPlausibleModel(column_names=feature_cols)
        curve = compute_feature_sensitivity(
            model, df, "radiation_anomaly_lag_1",
            feature_cols=feature_cols, n_points=5,
        )
        assert len(curve["feature_values"]) == 5

    def test_missing_feature_raises(self) -> None:
        df, feature_cols = make_audit_test_data()
        model = MockPhysicallyPlausibleModel(column_names=feature_cols)
        with pytest.raises(ValueError, match="not in DataFrame"):
            compute_feature_sensitivity(
                model, df, "nonexistent_feature",
                feature_cols=feature_cols,
            )

    def test_deterministic_with_seed(self) -> None:
        df, feature_cols = make_audit_test_data()
        # Use noise=0 for strict determinism
        model = MockPhysicallyPlausibleModel(noise=0.0, column_names=feature_cols, seed=42)
        c1 = compute_feature_sensitivity(
            model, df, "radiation_anomaly_lag_1",
            feature_cols=feature_cols, seed=42,
        )
        c2 = compute_feature_sensitivity(
            model, df, "radiation_anomaly_lag_1",
            feature_cols=feature_cols, seed=42,
        )
        np.testing.assert_array_almost_equal(c1["mean_prediction"], c2["mean_prediction"])


# ---------------------------------------------------------------------------
# Regional sensitivity
# ---------------------------------------------------------------------------


class TestRegionalSensitivity:
    def test_output_has_regions(self) -> None:
        df, feature_cols = make_audit_test_data()
        model = MockPhysicallyPlausibleModel(column_names=feature_cols)
        reg = compute_regional_sensitivity(
            model, df, "radiation_anomaly_lag_1",
            feature_cols=feature_cols, n_points=5,
        )
        assert "region" in reg.columns
        assert reg["region"].nunique() == 3
        assert "mean_prediction" in reg.columns

    def test_all_regions_present(self) -> None:
        df, feature_cols = make_audit_test_data()
        model = MockPhysicallyPlausibleModel(column_names=feature_cols)
        reg = compute_regional_sensitivity(
            model, df, "soil_moisture_anomaly_lag_1",
            feature_cols=feature_cols, n_points=5,
        )
        assert set(reg["region"].unique()) == {"Sahara", "Amazon", "East China"}


# ---------------------------------------------------------------------------
# Monotonic trend
# ---------------------------------------------------------------------------


class TestMonotonicTrend:
    def test_increasing_trend(self) -> None:
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1  # perfect increasing
        result = check_monotonic_trend(x, y)
        assert result["direction"] == "increasing"
        assert result["spearman_rho"] == pytest.approx(1.0)
        assert result["is_significant"] is True

    def test_decreasing_trend(self) -> None:
        x = np.linspace(0, 10, 50)
        y = -3 * x + 5  # perfect decreasing
        result = check_monotonic_trend(x, y)
        assert result["direction"] == "decreasing"
        assert result["spearman_rho"] == pytest.approx(-1.0)

    def test_flat_trend(self) -> None:
        x = np.linspace(0, 10, 50)
        y = np.ones(50)  # constant
        result = check_monotonic_trend(x, y)
        assert result["direction"] == "flat"
        assert result["monotonic"] is False

    def test_noisy_increasing(self) -> None:
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        y = 1.5 * x + rng.normal(0, 2, size=100)
        result = check_monotonic_trend(x, y)
        assert result["direction"] == "increasing"
        assert result["is_significant"] is True


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------


class TestIsConsistent:
    def test_positive_expected_matches_increasing(self) -> None:
        trend = {"direction": "increasing", "is_significant": True}
        assert is_consistent_with_expectation(trend, "positive") is True

    def test_positive_expected_mismatch(self) -> None:
        trend = {"direction": "decreasing", "is_significant": True}
        assert is_consistent_with_expectation(trend, "positive") is False

    def test_not_significant_returns_false(self) -> None:
        trend = {"direction": "increasing", "is_significant": False}
        assert is_consistent_with_expectation(trend, "positive") is False

    def test_negative_expected_matches_decreasing(self) -> None:
        trend = {"direction": "decreasing", "is_significant": True}
        assert is_consistent_with_expectation(trend, "negative") is True


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------


class TestRunPhysicalAudit:
    def test_full_audit_with_plausible_model(self) -> None:
        df, feature_cols = make_audit_test_data(n_samples=300)
        model = MockPhysicallyPlausibleModel(column_names=feature_cols)

        with tempfile.TemporaryDirectory() as tmp:
            result = run_physical_audit(
                model=model,
                model_name="mock_plausible",
                df=df,
                feature_cols=feature_cols,
                output_dir=tmp,
            )
            # Check summary structure
            assert "consistency_score" in result
            assert "n_features_audited" in result
            assert result["model_name"] == "mock_plausible"

            # Files created
            out = Path(tmp)
            assert (out / "consistency_summary.json").exists()
            assert (out / "physical_consistency_report.md").exists()
            assert (out / "regional_sensitivity.csv").exists()

            # PDP plots
            for feat in ["radiation_anomaly_lag_1", "soil_moisture_anomaly_lag_1"]:
                assert (out / f"pdp_{feat}.png").exists()

    def test_broken_model_gets_low_score(self) -> None:
        df, feature_cols = make_audit_test_data(n_samples=200)
        model = MockBrokenModel()

        with tempfile.TemporaryDirectory() as tmp:
            result = run_physical_audit(
                model=model,
                model_name="mock_broken",
                df=df,
                feature_cols=feature_cols,
                output_dir=tmp,
            )
            # A constant model should have flat trends → 0 consistency
            assert result["consistency_score"] == 0.0

    def test_missing_features_handled_gracefully(self) -> None:
        """If no audit features are in the data, audit should return gracefully."""
        df = pd.DataFrame({"x": [1, 2, 3], "region": ["a", "b", "c"]})
        model = MockPhysicallyPlausibleModel(column_names=["x"])

        with tempfile.TemporaryDirectory() as tmp:
            result = run_physical_audit(
                model=model,
                model_name="test",
                df=df,
                feature_cols=["x"],
                output_dir=tmp,
            )
            assert "error" in result

    def test_report_contains_expected_sections(self) -> None:
        df, feature_cols = make_audit_test_data(n_samples=200)
        model = MockPhysicallyPlausibleModel(column_names=feature_cols)

        with tempfile.TemporaryDirectory() as tmp:
            result = run_physical_audit(
                model=model,
                model_name="mock_plausible",
                df=df,
                feature_cols=feature_cols,
                output_dir=tmp,
            )
            report_path = Path(result["report_path"])
            report = report_path.read_text()

            assert "Physical Consistency Audit Report" in report
            assert "## Purpose" in report
            assert "## Feature-by-Feature Results" in report
            assert "## Regional Findings" in report
            assert "## Limitations" in report
            assert "not causal" in report.lower()
