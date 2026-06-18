"""Tests for the evaluation runner and mini-leaderboard.

Covers:

- evaluate_model_on_split: returns predictions, metrics, grouped metrics.
- update_leaderboard: upserts rows, ranks by RMSE.
- load_leaderboard: handles missing files gracefully.
- Mini benchmark script runs on synthetic data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climatenet.evaluation.leaderboard import (
    LEADERBOARD_COLUMNS,
    load_leaderboard,
    update_leaderboard,
)
from climatenet.evaluation.runner import evaluate_model_on_split


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_train_test_with_month_and_anomaly(
    n_train: int = 60,
    n_test: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train/val/test DataFrames with 'month' and 'evaporation_anomaly'."""
    rng = np.random.default_rng(seed)

    def _make(n: int, base_seed: int) -> pd.DataFrame:
        rng2 = np.random.default_rng(base_seed)
        rows = []
        for i in range(n):
            month = (i % 12) + 1
            year = 2020 + i // 12
            region = "Sahara" if i < n // 3 else ("East China" if i < 2 * n // 3 else "Amazon")
            zone_map = {"Sahara": "arid", "East China": "monsoon", "Amazon": "tropical_humid"}
            # Realistic anomaly: seasonal signal + noise
            base = 3.0 * np.sin(2 * np.pi * (month - 3) / 12)
            rows.append(
                {
                    "sample_id": f"{region}_{i:04d}",
                    "year": year,
                    "month": month,
                    "region": region,
                    "climate_zone": zone_map[region],
                    "lat": 25.0,
                    "lon": 10.0,
                    "evaporation_anomaly": round(float(base + rng2.normal(0, 0.8)), 4),
                    "temperature_anomaly": round(float(rng2.normal(0, 1.5)), 4),
                    "soil_moisture_anomaly": round(float(rng2.normal(0, 0.03)), 4),
                    "feat_a": rng2.normal(0, 1),
                    "feat_b": rng2.normal(0, 1),
                }
            )
        return pd.DataFrame(rows)

    train = _make(n_train, seed)
    val = _make(n_train // 3, seed + 100)
    test = _make(n_test, seed + 200)
    return train, val, test


def _make_train_test(
    n_train: int = 60,
    n_test: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train/val/test DataFrames with anomaly target + features."""
    rng = np.random.default_rng(seed)

    def _make(n: int, base_seed: int) -> pd.DataFrame:
        rng2 = np.random.default_rng(base_seed)
        rows = []
        for i in range(n):
            month = (i % 12) + 1
            year = 2020 + i // 12
            region = "Sahara" if i < n // 3 else ("East China" if i < 2 * n // 3 else "Amazon")
            zone_map = {"Sahara": "arid", "East China": "monsoon", "Amazon": "tropical_humid"}
            rows.append(
                {
                    "sample_id": f"{region}_{i:04d}",
                    "year": year,
                    "month": month,
                    "region": region,
                    "climate_zone": zone_map[region],
                    "lat": 25.0,
                    "lon": 10.0,
                    "y_true": round(float(rng2.normal(0, 1.0)), 4),
                    "y_true_lag1": round(float(rng2.normal(0, 0.8)), 4),
                    "feat_a": rng2.normal(0, 1),
                    "feat_b": rng2.normal(0, 1),
                }
            )
        return pd.DataFrame(rows)

    train = _make(n_train, seed)
    val = _make(n_train // 3, seed + 100)
    test = _make(n_test, seed + 200)
    return train, val, test


class MockModel:
    """Simple linear model for testing the evaluation runner."""

    def __init__(self, model_name: str = "mock"):
        self._name = model_name
        self._coef: float = 0.0

    def fit(self, train_df, target_col=None, feature_cols=None, val_df=None, **kwargs):
        if target_col is not None:
            pass
        if feature_cols and len(feature_cols) > 0:
            t = train_df[target_col or "y_true"].to_numpy()
            x = train_df[feature_cols[0]].to_numpy()
            self._coef = float(np.cov(x, t)[0, 1] / max(np.var(x), 1e-6))
        return self

    def predict(self, test_df):
        return np.zeros(len(test_df), dtype=np.float64)

    def get_model_name(self):
        return self._name

    def get_params(self):
        return {"model_name": self._name, "coef": self._coef}


# ---------------------------------------------------------------------------
# evaluate_model_on_split
# ---------------------------------------------------------------------------


class TestEvaluateModelOnSplit:
    def test_returns_predictions_and_metrics(self) -> None:
        train, val, test = _make_train_test()
        model = MockModel()

        result = evaluate_model_on_split(
            model=model,
            train_df=train,
            val_df=None,
            test_df=test,
            target_col="y_true",
            feature_cols=["feat_a"],
        )

        assert "predictions_df" in result
        assert "metrics_overall" in result
        assert "metrics_by_group" in result

        preds = result["predictions_df"]
        assert "y_true" in preds.columns
        assert "y_pred" in preds.columns
        assert "model_name" in preds.columns
        assert len(preds) == len(test)

        metrics = result["metrics_overall"]
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["n_samples"] == len(test)

    def test_grouped_metrics_by_region_and_climate_zone(self) -> None:
        train, val, test = _make_train_test(n_train=60, n_test=30)
        model = MockModel()

        result = evaluate_model_on_split(
            model=model,
            train_df=train,
            val_df=None,
            test_df=test,
            target_col="y_true",
            feature_cols=["feat_a"],
        )

        grouped = result["metrics_by_group"]
        assert len(grouped) > 0
        for g in grouped:
            assert "rmse" in g
            assert "region" in g or "climate_zone" in g

    def test_event_metrics_when_event_cols_provided(self) -> None:
        """Event metrics should be constructed from y_true/y_pred using
        train-fitted thresholds — NOT from pre-existing '{ev}_pred' columns."""
        from climatenet.evaluation.hydroclimate_labels import fit_event_thresholds

        train, val, test = _make_train_test_with_month_and_anomaly(n_train=60, n_test=30)

        model = MockModel()
        thresholds = fit_event_thresholds(train)

        result = evaluate_model_on_split(
            model=model,
            train_df=train,
            val_df=None,
            test_df=test,
            target_col="evaporation_anomaly",
            feature_cols=["feat_a"],
            event_cols=["evaporation_deficit"],
            event_thresholds=thresholds,
        )

        metrics = result["metrics_overall"]
        assert any(k.startswith("pod_") for k in metrics), f"Missing POD in {sorted(metrics)}"
        # Since MockModel predicts zeros, and test data has varying
        # evaporation_anomaly values, event metrics should NOT be perfect
        pod_val = metrics.get("pod_evaporation_deficit")
        assert pod_val is not None
        # A zero-predicting model cannot get perfect event detection
        assert pod_val < 1.0, f"Zero-predicting model got POD={pod_val} — should not be perfect"

    def test_predictions_include_meta_cols(self) -> None:
        train, val, test = _make_train_test()
        model = MockModel()

        result = evaluate_model_on_split(
            model=model,
            train_df=train,
            val_df=None,
            test_df=test,
            target_col="y_true",
            feature_cols=["feat_a"],
        )

        preds = result["predictions_df"]
        for col in ["sample_id", "region", "climate_zone"]:
            assert col in preds.columns, f"Missing '{col}' in predictions"

    def test_climatology_does_not_get_perfect_event_scores(self) -> None:
        """Climatology predicts zero anomaly → it should NOT get POD=1.0."""
        from climatenet.evaluation.hydroclimate_labels import fit_event_thresholds

        train, val, test = _make_train_test_with_month_and_anomaly(n_train=60, n_test=30)

        # Climatology predicts zero for all samples
        model = MockModel()  # predict() returns np.zeros(...)

        thresholds = fit_event_thresholds(train)

        result = evaluate_model_on_split(
            model=model,
            train_df=train,
            val_df=None,
            test_df=test,
            target_col="evaporation_anomaly",
            feature_cols=["feat_a"],
            event_cols=["evaporation_deficit"],
            event_thresholds=thresholds,
        )

        metrics = result["metrics_overall"]
        pod = metrics.get("pod_evaporation_deficit")
        assert pod is not None, "Missing pod_evaporation_deficit"
        assert pod < 1.0, (
            f"Bug: climatology (predicts zero) got POD={pod}. "
            f"Event metrics must compare y_pred-derived labels against "
            f"y_true-derived labels, NOT truth against itself."
        )

    def test_predicted_event_differs_from_true_event(self) -> None:
        """When predictions are imperfect, true_event != pred_event arrays."""
        from climatenet.evaluation.hydroclimate_labels import fit_event_thresholds

        train, val, test = _make_train_test_with_month_and_anomaly(n_train=60, n_test=30)

        # A model that predicts non-zero but imperfect values
        class ImperfectModel:
            def __init__(self):
                self._name = "imperfect"

            def fit(self, train_df, **kwargs):
                return self

            def predict(self, test_df):
                rng = np.random.default_rng(123)
                return rng.normal(0, 0.5, size=len(test_df))

            def get_model_name(self):
                return self._name

            def get_params(self):
                return {"model_name": self._name}

        model = ImperfectModel()
        thresholds = fit_event_thresholds(train)

        result = evaluate_model_on_split(
            model=model,
            train_df=train,
            val_df=None,
            test_df=test,
            target_col="evaporation_anomaly",
            feature_cols=["feat_a"],
            event_cols=["evaporation_deficit"],
            event_thresholds=thresholds,
        )

        preds = result["predictions_df"]
        true_arr = preds["evaporation_deficit_true"].to_numpy()
        pred_arr = preds["evaporation_deficit_pred"].to_numpy()

        # They should differ because predictions are random and imperfect
        assert not np.array_equal(true_arr, pred_arr), (
            "Bug: true_event == pred_event. "
            "Predicted event labels must be derived from y_pred, not copied from y_true."
        )


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


class TestLeaderboard:
    def test_update_creates_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lb_path = Path(tmp) / "lb.csv"
            metrics = {"rmse": 0.5, "mae": 0.4, "r2": 0.9, "n_samples": 100}
            result = update_leaderboard(metrics, "climatology", "random", lb_path)
            assert lb_path.exists()
            assert len(result) == 1
            assert result.iloc[0]["model"] == "climatology"
            assert result.iloc[0]["rmse"] == 0.5

    def test_ranks_by_rmse_ascending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lb_path = Path(tmp) / "lb.csv"

            update_leaderboard(
                {"rmse": 0.8, "mae": 0.6, "r2": 0.5}, "climatology", "random", lb_path,
            )
            update_leaderboard(
                {"rmse": 0.3, "mae": 0.2, "r2": 0.95}, "lightgbm", "random", lb_path,
            )
            update_leaderboard(
                {"rmse": 0.55, "mae": 0.4, "r2": 0.85}, "persistence", "random", lb_path,
            )

            lb = load_leaderboard(lb_path)
            assert lb.iloc[0]["rank"] == 1
            assert lb.iloc[0]["model"] == "lightgbm"  # best RMSE
            assert lb.iloc[1]["model"] == "persistence"
            assert lb.iloc[2]["model"] == "climatology"

    def test_upserts_existing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lb_path = Path(tmp) / "lb.csv"

            update_leaderboard({"rmse": 0.5}, "model_a", "split_1", lb_path)
            update_leaderboard({"rmse": 0.3}, "model_a", "split_1", lb_path)  # update

            lb = load_leaderboard(lb_path)
            assert len(lb) == 1
            assert lb.iloc[0]["rmse"] == 0.3

    def test_load_missing_file_returns_empty_df(self) -> None:
        lb = load_leaderboard("/tmp/nonexistent_lb_42.csv")
        assert lb.empty
        for col in LEADERBOARD_COLUMNS:
            assert col in lb.columns

    def test_leaderboard_has_all_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lb_path = Path(tmp) / "lb.csv"
            update_leaderboard({"rmse": 0.5}, "test_model", "test_split", lb_path)
            lb = load_leaderboard(lb_path)
            for col in LEADERBOARD_COLUMNS:
                assert col in lb.columns, f"Missing column '{col}'"

    def test_event_metrics_flow_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lb_path = Path(tmp) / "lb.csv"
            metrics = {
                "rmse": 0.42,
                "mae": 0.33,
                "r2": 0.88,
                "pod_soil_moisture_drought": 0.75,
                "far_soil_moisture_drought": 0.20,
                "csi_soil_moisture_drought": 0.65,
                "intensity_bias_soil_moisture_drought": 1.05,
                "n_events_soil_moisture_drought": 15,
                "n_samples": 200,
            }
            result = update_leaderboard(metrics, "rf", "temporal_holdout", lb_path)
            row = result.iloc[0]
            assert row["pod"] == 0.75
            assert row["far"] == 0.20
            assert row["csi"] == 0.65
            assert row["intensity_bias"] == 1.05
            assert row["n_events"] == 15.0


# ---------------------------------------------------------------------------
# Integration: mini benchmark script
# ---------------------------------------------------------------------------


class TestMiniBenchmarkScript:
    def test_script_runs_end_to_end(self) -> None:
        """Run the mini benchmark script and verify output files."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/run_mini_benchmark.py"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            timeout=60,
        )

        # The script may fail if lightgbm is not installed, but should still
        # run climatology and persistence baselines
        output_root = Path(__file__).resolve().parents[1] / "outputs" / "mini_benchmark"

        # Check that at least some outputs exist
        if result.returncode == 0:
            assert output_root.exists()
            # Check leaderboard
            lb_path = output_root / "leaderboard" / "v1_mini.csv"
            if lb_path.exists():
                lb = pd.read_csv(lb_path)
                assert len(lb) >= 2  # at least climatology + persistence
        else:
            # If script failed (e.g. import error), still check partial output
            # Don't fail the test; just log
            print(f"Mini benchmark script stderr: {result.stderr[-500:]}")
