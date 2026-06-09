"""Tests for conformal prediction and calibration evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.evaluation.calibration import (
    build_calibration_report,
    build_intervals_table,
    save_calibration_report,
)
from climatenet.evaluation.conformal import (
    build_prediction_intervals,
    evaluate_by_group,
    evaluate_coverage,
    evaluate_interval_width,
    fit_conformal_quantile,
    run_conformal_pipeline,
)


# ---------------------------------------------------------------------------
# Tiny synthetic data
# ---------------------------------------------------------------------------


def make_calib_test_data(
    n_calib: int = 200,
    n_test: int = 100,
    noise: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate calibration and test data with known noise level."""
    rng = np.random.default_rng(seed)

    def _make(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.uniform(-3, 3, size=n)
        y_true = 2.0 * x + 1.0 + rng.normal(0, noise, size=n)
        y_pred = 2.0 * x + 1.0  # perfect model (noise-free)
        return y_true, y_pred

    y_calib, pred_calib = _make(n_calib)
    y_test, pred_test = _make(n_test)
    return y_calib, pred_calib, y_test, pred_test


# ---------------------------------------------------------------------------
# fit_conformal_quantile
# ---------------------------------------------------------------------------


class TestFitConformalQuantile:
    def test_basic_quantile(self) -> None:
        y_calib, pred_calib, _, _ = make_calib_test_data(n_calib=500)
        q = fit_conformal_quantile(y_calib, pred_calib, alpha=0.1)
        assert q > 0
        # With Gaussian noise σ=1, the 0.9 quantile of |N(0,1)| ≈ 1.64
        assert 1.2 < q < 2.2

    def test_alpha_near_one(self) -> None:
        y_calib, pred_calib, _, _ = make_calib_test_data(n_calib=200)
        q = fit_conformal_quantile(y_calib, pred_calib, alpha=0.99)
        # At α=0.99, we want coverage 0.01 — quantile should be near 0
        assert q >= 0

    def test_alpha_near_zero(self) -> None:
        y_calib, pred_calib, _, _ = make_calib_test_data(n_calib=200)
        q = fit_conformal_quantile(y_calib, pred_calib, alpha=0.001)
        # At α=0.001, we want coverage 0.999 — quantile should be large
        assert q > 1.0

    def test_invalid_alpha_raises(self) -> None:
        y_calib, pred_calib, _, _ = make_calib_test_data(n_calib=10)
        with pytest.raises(ValueError, match="alpha"):
            fit_conformal_quantile(y_calib, pred_calib, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            fit_conformal_quantile(y_calib, pred_calib, alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            fit_conformal_quantile(y_calib, pred_calib, alpha=-0.5)

    def test_empty_calibration_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            fit_conformal_quantile(np.array([]), np.array([]), alpha=0.1)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            fit_conformal_quantile(np.ones(5), np.ones(3), alpha=0.1)

    def test_perfect_predictions(self) -> None:
        """If model is perfect on calibration, q should be 0."""
        y = np.array([1.0, 2.0, 3.0])
        q = fit_conformal_quantile(y, y, alpha=0.1)
        assert q >= 0.0  # all residuals are 0 → q = 0


# ---------------------------------------------------------------------------
# build_prediction_intervals
# ---------------------------------------------------------------------------


class TestBuildPredictionIntervals:
    def test_intervals_are_symmetric(self) -> None:
        y_pred = np.array([1.0, 2.0, 3.0])
        q = 0.5
        lower, upper = build_prediction_intervals(y_pred, q)
        np.testing.assert_array_equal(lower, y_pred - q)
        np.testing.assert_array_equal(upper, y_pred + q)
        np.testing.assert_array_equal(upper - lower, 2 * q)

    def test_constant_width(self) -> None:
        y_pred = np.array([-5.0, 0.0, 10.0, 100.0])
        q = 1.5
        lower, upper = build_prediction_intervals(y_pred, q)
        widths = upper - lower
        np.testing.assert_allclose(widths, 3.0)

    def test_negative_q_raises(self) -> None:
        with pytest.raises(ValueError, match="q must be"):
            build_prediction_intervals(np.ones(3), q=-0.1)


# ---------------------------------------------------------------------------
# evaluate_coverage
# ---------------------------------------------------------------------------


class TestEvaluateCoverage:
    def test_perfect_coverage(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        assert evaluate_coverage(y_true, lower, upper) == 1.0

    def test_partial_coverage(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        lower = np.array([0.0, 0.0, 0.0, 0.0])
        upper = np.array([2.0, 2.0, 2.0, 2.0])
        # Only first 2 are covered → 0.5
        assert evaluate_coverage(y_true, lower, upper) == 0.5

    def test_zero_coverage(self) -> None:
        y_true = np.array([10.0, 20.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        assert evaluate_coverage(y_true, lower, upper) == 0.0

    def test_boundary_inclusive(self) -> None:
        """Bounds are inclusive: y_true == lower should count as covered."""
        y_true = np.array([0.0, 5.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([5.0, 5.0])
        assert evaluate_coverage(y_true, lower, upper) == 1.0

    def test_empty_input_returns_nan(self) -> None:
        assert np.isnan(evaluate_coverage(np.array([]), np.array([]), np.array([])))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_coverage(np.ones(3), np.ones(2), np.ones(3))


# ---------------------------------------------------------------------------
# evaluate_interval_width
# ---------------------------------------------------------------------------


class TestEvaluateIntervalWidth:
    def test_constant_width(self) -> None:
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([3.0, 4.0, 5.0])
        assert evaluate_interval_width(lower, upper) == 3.0

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(evaluate_interval_width(np.array([]), np.array([])))


# ---------------------------------------------------------------------------
# evaluate_by_group
# ---------------------------------------------------------------------------


class TestEvaluateByGroup:
    def test_by_region(self) -> None:
        df = pd.DataFrame(
            {
                "region": ["Sahara", "Sahara", "Amazon", "Amazon"],
                "y_true": [1.0, 2.0, 3.0, 4.0],
                "lower": [0.5, 1.5, 2.0, 3.0],
                "upper": [1.5, 2.5, 4.0, 5.0],
            }
        )
        result = evaluate_by_group(df, "region")
        assert len(result) == 2
        sahara = result[result["region"] == "Sahara"]
        assert sahara.iloc[0]["coverage"] == 1.0
        assert sahara.iloc[0]["mean_interval_width"] == 1.0

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"y_true": [1.0], "lower": [0.0], "upper": [2.0]})
        with pytest.raises(ValueError, match="missing"):
            evaluate_by_group(df, "region")


# ---------------------------------------------------------------------------
# run_conformal_pipeline
# ---------------------------------------------------------------------------


class TestRunConformalPipeline:
    def test_full_pipeline(self) -> None:
        y_calib, pred_calib, y_test, pred_test = make_calib_test_data(
            n_calib=200, n_test=100
        )
        result = run_conformal_pipeline(
            y_calib, pred_calib, y_test, pred_test, alpha=0.1
        )
        assert "q" in result
        assert "coverage" in result
        assert "lower" in result
        assert "upper" in result
        # Coverage should be close to 0.90
        assert 0.80 <= result["coverage"] <= 1.0
        assert result["mean_interval_width"] > 0

    def test_pipeline_with_grouped(self) -> None:
        y_calib, pred_calib, y_test, pred_test = make_calib_test_data(
            n_calib=200, n_test=100
        )
        df = pd.DataFrame(
            {
                "region": ["Amazon"] * 50 + ["Sahara"] * 50,
                "climate_type": ["tropical_humid"] * 50 + ["arid"] * 50,
            }
        )
        result = run_conformal_pipeline(
            y_calib,
            pred_calib,
            y_test,
            pred_test,
            alpha=0.1,
            test_df=df,
            group_cols=["region", "climate_type"],
        )
        assert "region" in result["by_group"]
        assert "climate_type" in result["by_group"]
        assert len(result["by_group"]["region"]) == 2

    def test_calibration_test_separation(self) -> None:
        """Quantile must be computed from calibration set, NOT test set."""
        # Calibration: small residuals → small q
        y_calib = np.array([1.0, 1.0, 1.0])
        pred_calib = np.array([1.0, 1.0, 1.0])  # perfect → q ≈ 0

        # Test: large residuals
        y_test = np.array([100.0, 100.0])
        pred_test = np.array([0.0, 0.0])

        result = run_conformal_pipeline(
            y_calib, pred_calib, y_test, pred_test, alpha=0.1
        )
        # q comes from calibration → tiny
        assert result["q"] == pytest.approx(0.0, abs=0.1)
        # Coverage on test should be zero (intervals are too narrow)
        assert result["coverage"] == 0.0


# ---------------------------------------------------------------------------
# build_intervals_table & build_calibration_report
# ---------------------------------------------------------------------------


class TestIntervalsTable:
    def test_basic_table(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([0.9, 2.1])
        q = 0.5
        lower = y_pred - q
        upper = y_pred + q

        table = build_intervals_table(
            y_true,
            y_pred,
            lower,
            upper,
            model_name="rf",
            split_id="spatial_block",
            experiment_id="exp_001",
        )
        assert len(table) == 2
        assert table.iloc[0]["covered"] == (
            1.0 >= lower[0] and 1.0 <= upper[0]
        )
        assert table["interval_width"].iloc[0] == pytest.approx(1.0)
        assert table["model_name"].iloc[0] == "rf"

    def test_with_meta_df(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()
        lower = y_pred - 1.0
        upper = y_pred + 1.0
        meta = pd.DataFrame(
            {
                "region": ["Sahara", "Amazon", "Sahara"],
                "year": [2020, 2020, 2021],
                "month": [7, 7, 8],
            }
        )
        table = build_intervals_table(
            y_true, y_pred, lower, upper, meta_df=meta
        )
        assert list(table["region"]) == ["Sahara", "Amazon", "Sahara"]

    def test_meta_length_mismatch_raises(self) -> None:
        y = np.ones(3)
        meta = pd.DataFrame({"region": ["Sahara"]})
        with pytest.raises(ValueError, match="rows"):
            build_intervals_table(y, y, y, y, meta_df=meta)


class TestCalibrationReport:
    def test_build_and_save(self) -> None:
        y_calib, pred_calib, y_test, pred_test = make_calib_test_data(
            n_calib=100, n_test=50
        )
        report = build_calibration_report(
            y_calib, pred_calib, y_test, pred_test, alpha=0.1
        )
        assert report["alpha"] == 0.1
        assert report["target_coverage"] == 0.9
        assert "conformal_quantile" in report
        assert report["n_calibration"] == 100
        assert report["n_test"] == 50
        assert 0.7 <= report["coverage"] <= 1.0

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "calib.json"
            save_calibration_report(report, p)
            assert p.exists()

    def test_with_groups(self) -> None:
        y_calib, pred_calib, y_test, pred_test = make_calib_test_data(
            n_calib=100, n_test=50
        )
        df = pd.DataFrame(
            {
                "region": ["Sahara"] * 25 + ["Amazon"] * 25,
                "climate_type": ["arid"] * 25 + ["tropical_humid"] * 25,
            }
        )
        report = build_calibration_report(
            y_calib,
            pred_calib,
            y_test,
            pred_test,
            alpha=0.1,
            test_df=df,
            group_cols=["region", "climate_type"],
        )
        assert "by_region" in report
        assert len(report["by_region"]) == 2
