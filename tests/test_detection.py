"""Tests for binary event detection metrics (POD, FAR, CSI, intensity_bias)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climatenet.evaluation.detection import (
    compute_csi,
    compute_event_detection_table,
    compute_far,
    compute_intensity_bias,
    compute_pod,
)
from climatenet.evaluation.hydroclimate_labels import ALL_EVENT_TYPES


# ---------------------------------------------------------------------------
# POD
# ---------------------------------------------------------------------------


class TestPOD:
    def test_perfect(self) -> None:
        y = np.array([True, True, False, False])
        result = compute_pod(y, y)
        assert result["value"] == 1.0
        assert "warning" not in result

    def test_misses_some(self) -> None:
        # 3 observed events, 2 correctly predicted
        y_true = np.array([True, True, True, False, False])
        y_pred = np.array([True, True, False, False, False])
        result = compute_pod(y_true, y_pred)
        assert result["value"] == pytest.approx(2 / 3)

    def test_misses_all(self) -> None:
        y_true = np.array([True, True, True])
        y_pred = np.array([False, False, False])
        result = compute_pod(y_true, y_pred)
        assert result["value"] == 0.0

    def test_no_observed_events_returns_nan(self) -> None:
        y_true = np.array([False, False, False])
        y_pred = np.array([True, False, False])
        result = compute_pod(y_true, y_pred)
        assert np.isnan(result["value"])
        assert "no observed events" in result["warning"]

    def test_empty_input_returns_nan(self) -> None:
        y_true = np.array([], dtype=bool)
        y_pred = np.array([], dtype=bool)
        result = compute_pod(y_true, y_pred)
        assert np.isnan(result["value"])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_pod(np.ones(3, dtype=bool), np.ones(2, dtype=bool))

    def test_2d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            compute_pod(np.ones((2, 3)), np.ones((2, 3)))

    def test_0_1_integers_accepted(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        result = compute_pod(y_true, y_pred)
        assert result["value"] == 0.5


# ---------------------------------------------------------------------------
# FAR
# ---------------------------------------------------------------------------


class TestFAR:
    def test_perfect(self) -> None:
        y_true = np.array([True, True, False, False])
        y_pred = np.array([True, True, False, False])
        result = compute_far(y_true, y_pred)
        assert result["value"] == 0.0

    def test_false_alarms(self) -> None:
        # 2 hits, 2 false alarms
        y_true = np.array([True, True, False, False])
        y_pred = np.array([True, True, True, True])
        result = compute_far(y_true, y_pred)
        assert result["value"] == pytest.approx(2 / 4)

    def test_all_false_alarms(self) -> None:
        y_true = np.array([False, False, False])
        y_pred = np.array([True, True, True])
        result = compute_far(y_true, y_pred)
        assert result["value"] == 1.0

    def test_no_predicted_events_returns_nan(self) -> None:
        y_true = np.array([True, True, False])
        y_pred = np.array([False, False, False])
        result = compute_far(y_true, y_pred)
        assert np.isnan(result["value"])
        assert "no predicted events" in result["warning"]

    def test_no_events_at_all_returns_nan(self) -> None:
        y_true = np.array([False, False])
        y_pred = np.array([False, False])
        result = compute_far(y_true, y_pred)
        assert np.isnan(result["value"])
        # TP=0, FP=0 → denominator 0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_far(np.ones(5, dtype=bool), np.ones(3, dtype=bool))


# ---------------------------------------------------------------------------
# CSI
# ---------------------------------------------------------------------------


class TestCSI:
    def test_perfect(self) -> None:
        y_true = np.array([True, True, False, False])
        y_pred = np.array([True, True, False, False])
        result = compute_csi(y_true, y_pred)
        assert result["value"] == 1.0

    def test_partial_match(self) -> None:
        # TP=2, FP=1, FN=1
        y_true = np.array([True, True, True, False])
        y_pred = np.array([True, True, False, True])
        result = compute_csi(y_true, y_pred)
        assert result["value"] == pytest.approx(2 / 4)

    def test_no_hits(self) -> None:
        y_true = np.array([True, True])
        y_pred = np.array([False, False])
        result = compute_csi(y_true, y_pred)
        assert result["value"] == 0.0

    def test_no_events_at_all_returns_nan(self) -> None:
        y_true = np.array([False, False])
        y_pred = np.array([False, False])
        result = compute_csi(y_true, y_pred)
        assert np.isnan(result["value"])
        assert "no events in either" in result["warning"]


# ---------------------------------------------------------------------------
# Intensity bias
# ---------------------------------------------------------------------------


class TestIntensityBias:
    def test_perfect_prediction(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_label = np.array([True, True, False, False, False])
        result = compute_intensity_bias(y_true, y_pred, y_label)
        # Only first 2 samples: mean(y_true)=1.5, mean(y_pred)=1.5 → ratio=1.0
        assert result["value"] == pytest.approx(1.0)

    def test_overprediction(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 4.0, 3.0])
        y_label = np.array([True, True, False])
        result = compute_intensity_bias(y_true, y_pred, y_label)
        # Event samples: y_true=[1,2] → mean=1.5; y_pred=[2,4] → mean=3.0; ratio=2.0
        assert result["value"] == pytest.approx(2.0)

    def test_underprediction(self) -> None:
        y_true = np.array([2.0, 4.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        y_label = np.array([True, True, False])
        result = compute_intensity_bias(y_true, y_pred, y_label)
        assert result["value"] == pytest.approx(0.5)

    def test_no_observed_events_returns_nan(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        y_label = np.array([False, False, False])
        result = compute_intensity_bias(y_true, y_pred, y_label)
        assert np.isnan(result["value"])
        assert "no observed events" in result["warning"]

    def test_zero_mean_observed_returns_nan(self) -> None:
        y_true = np.array([0.0, 0.0, 1.0])
        y_pred = np.array([1.0, -1.0, 1.0])
        y_label = np.array([True, True, False])
        result = compute_intensity_bias(y_true, y_pred, y_label)
        assert np.isnan(result["value"])
        assert "mean observed value" in result["warning"]

    def test_n_event_samples_counted(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 4.0, 6.0, 8.0])
        y_label = np.array([True, True, True, False])
        result = compute_intensity_bias(y_true, y_pred, y_label)
        assert result["n_event_samples"] == 3

    def test_non_event_samples_ignored(self) -> None:
        """Samples where y_label is False must not affect the bias."""
        y_true = np.array([1.0, 2.0, 999.0])
        y_pred = np.array([2.0, 4.0, -999.0])
        y_label = np.array([True, True, False])
        result = compute_intensity_bias(y_true, y_pred, y_label)
        # Only first 2: mean(y_true)=1.5, mean(y_pred)=3.0 → 2.0
        assert result["value"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Batch evaluation table
# ---------------------------------------------------------------------------


class TestComputeEventDetectionTable:
    def _make_results_df(
        self,
        y_true_vals: list[float],
        y_pred_vals: list[float],
        obs_labels: dict[str, list[bool]],
        pred_labels: dict[str, list[bool]],
    ) -> pd.DataFrame:
        """Build a minimal results DataFrame for testing."""
        data = {"y_true": y_true_vals, "y_pred": y_pred_vals}
        for ev in obs_labels:
            data[ev] = obs_labels[ev]
            data[f"{ev}_pred"] = pred_labels[ev]
        return pd.DataFrame(data)

    def test_basic_table_shape(self) -> None:
        n = 50
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, n)
        y_pred = y_true + rng.normal(0, 0.5, n)

        obs = {}
        pred = {}
        for ev in ALL_EVENT_TYPES:
            obs[ev] = rng.random(n) < 0.15
            pred[ev] = rng.random(n) < 0.12

        df = self._make_results_df(y_true.tolist(), y_pred.tolist(), obs, pred)

        table = compute_event_detection_table(df)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 3  # 3 event types
        assert set(table["event_type"]) == set(ALL_EVENT_TYPES)
        assert "pod" in table.columns
        assert "far" in table.columns
        assert "csi" in table.columns
        assert "intensity_bias" in table.columns

    def test_missing_observed_column_raises(self) -> None:
        df = pd.DataFrame(
            {
                "y_true": [1.0, 2.0],
                "y_pred": [1.1, 1.9],
                "soil_moisture_drought": [True, False],
                "soil_moisture_drought_pred": [True, False],
                # Missing evaporation_deficit
                "evaporation_deficit_pred": [False, True],
                "compound_hot_dry": [False, False],
                "compound_hot_dry_pred": [False, False],
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            compute_event_detection_table(df)

    def test_missing_predicted_column_raises(self) -> None:
        df = pd.DataFrame(
            {
                "y_true": [1.0, 2.0],
                "y_pred": [1.1, 1.9],
                "soil_moisture_drought": [True, False],
                "soil_moisture_drought_pred": [True, False],
                "evaporation_deficit": [False, True],
                # Missing evaporation_deficit_pred
                "compound_hot_dry": [False, False],
                "compound_hot_dry_pred": [False, False],
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            compute_event_detection_table(df)

    def test_pod_far_csi_in_bounds(self) -> None:
        """All metric values should be in [0, 1] or NaN."""
        n = 100
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, n)
        y_pred = y_true + rng.normal(0, 0.3, n)

        obs = {}
        pred = {}
        for ev in ALL_EVENT_TYPES:
            obs[ev] = rng.random(n) < 0.15
            pred[ev] = rng.random(n) < 0.12

        df = self._make_results_df(y_true.tolist(), y_pred.tolist(), obs, pred)
        table = compute_event_detection_table(df)

        for col in ["pod", "far", "csi"]:
            for _, row in table.iterrows():
                val = row[col]
                assert (0.0 <= val <= 1.0) or np.isnan(val), f"{col}={val} out of bounds"

    def test_all_no_events_gives_nan_metrics(self) -> None:
        """When no events observed and none predicted, all metrics should be NaN."""
        df = self._make_results_df(
            [1.0, 2.0, 3.0],
            [1.1, 1.9, 3.1],
            {
                "soil_moisture_drought": [False, False, False],
                "evaporation_deficit": [False, False, False],
                "compound_hot_dry": [False, False, False],
            },
            {
                "soil_moisture_drought": [False, False, False],
                "evaporation_deficit": [False, False, False],
                "compound_hot_dry": [False, False, False],
            },
        )
        table = compute_event_detection_table(df)
        for col in ["pod", "far", "csi"]:
            assert table[col].isna().all()

    def test_custom_event_types(self) -> None:
        df = pd.DataFrame(
            {
                "y_true": [1.0, 2.0],
                "y_pred": [1.1, 1.9],
                "custom_event": [True, False],
                "custom_event_pred": [True, True],
            }
        )
        table = compute_event_detection_table(df, event_types=["custom_event"])
        assert len(table) == 1
        assert table.iloc[0]["event_type"] == "custom_event"


# ---------------------------------------------------------------------------
# Warning metadata propagation
# ---------------------------------------------------------------------------


class TestWarningMetadata:
    def test_pod_warning_in_table(self) -> None:
        df = pd.DataFrame(
            {
                "y_true": [1.0, 2.0, 3.0],
                "y_pred": [1.1, 1.9, 3.1],
                "soil_moisture_drought": [False, False, False],
                "soil_moisture_drought_pred": [True, False, False],
                "evaporation_deficit": [True, False, False],
                "evaporation_deficit_pred": [True, False, False],
                "compound_hot_dry": [False, False, False],
                "compound_hot_dry_pred": [False, False, False],
            }
        )
        table = compute_event_detection_table(df)
        # soil_moisture_drought: no observed → POD NaN + warning
        sm_row = table[table["event_type"] == "soil_moisture_drought"].iloc[0]
        assert np.isnan(sm_row["pod"])
        assert sm_row["pod_warning"] is not None

        # compound_hot_dry: no observed + no predicted → all NaN
        chd_row = table[table["event_type"] == "compound_hot_dry"].iloc[0]
        assert np.isnan(chd_row["pod"])
        assert np.isnan(chd_row["far"])
        assert np.isnan(chd_row["csi"])
