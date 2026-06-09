"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climatenet.evaluation.metrics import (
    _validate_inputs,
    evaluate_regression,
    mae,
    r2,
    rmse,
)
from climatenet.evaluation.ood_degradation import (
    compute_ood_degradation,
    compute_ood_degradation_table,
)
from climatenet.evaluation.skill_score import compute_skill_scores, skill_score


# ---------------------------------------------------------------------------
#  Primary metrics
# ---------------------------------------------------------------------------


class TestPrimaryMetrics:
    def test_mae_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_mae_positive(self) -> None:
        assert mae(np.array([0.0, 0.0]), np.array([1.0, 2.0])) == 1.5

    def test_rmse_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_positive(self) -> None:
        # preds off by [1, 2] → squared = [1, 4] → mean = 2.5 → sqrt ≈ 1.581
        result = rmse(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
        assert result == pytest.approx(np.sqrt(2.5))

    def test_r2_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_r2_constant_mean(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 2.0, 2.0])  # always predicts mean
        assert r2(y, pred) == pytest.approx(0.0, abs=1e-10)

    def test_r2_worse_than_mean(self) -> None:
        # y has variance; pred is far off → should be worse than mean → R² < 0
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        assert r2(y, pred) < 0.0

    def test_evaluate_regression(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        result = evaluate_regression(y, y)
        assert result == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}

    # --- input validation ---

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            mae(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_arrays_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mae(np.array([]), np.array([]))

    def test_nan_in_true_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            mae(np.array([1.0, np.nan]), np.array([2.0, 3.0]))

    def test_nan_in_pred_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            mae(np.array([1.0, 2.0]), np.array([np.nan, 3.0]))

    def test_2d_arrays_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            mae(np.ones((2, 3)), np.ones((2, 3)))

    def test_float_input_accepted(self) -> None:
        """Lists should be converted to arrays."""
        result = mae([1.0, 2.0], [1.0, 2.0])
        assert result == 0.0


# ---------------------------------------------------------------------------
#  Skill score
# ---------------------------------------------------------------------------


class TestSkillScore:
    def test_perfect_model_gives_skill_one(self) -> None:
        assert skill_score(model_rmse=0.0, baseline_rmse=1.0) == 1.0

    def test_equal_to_baseline_gives_zero(self) -> None:
        assert skill_score(model_rmse=2.0, baseline_rmse=2.0) == 0.0

    def test_better_than_baseline_positive(self) -> None:
        # RMSE 0.5 vs 1.0 → 50 % improvement
        ss = skill_score(model_rmse=0.5, baseline_rmse=1.0)
        assert ss == 0.5
        assert ss > 0

    def test_worse_than_baseline_negative(self) -> None:
        # RMSE 1.5 vs 1.0 → 50 % worse
        ss = skill_score(model_rmse=1.5, baseline_rmse=1.0)
        assert ss == -0.5
        assert ss < 0

    def test_baseline_zero_returns_nan(self) -> None:
        assert np.isnan(skill_score(model_rmse=0.5, baseline_rmse=0.0))

    def test_negative_model_rmse_raises(self) -> None:
        with pytest.raises(ValueError, match="model_rmse"):
            skill_score(model_rmse=-1.0, baseline_rmse=1.0)

    def test_negative_baseline_rmse_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_rmse"):
            skill_score(model_rmse=1.0, baseline_rmse=-1.0)

    def test_skill_vs_climatology_example(self) -> None:
        """If model RMSE = 0.42 and climatology RMSE = 0.63, skill = 1 - 0.42/0.63 ≈ 0.333."""
        ss = skill_score(0.42, 0.63)
        assert ss == pytest.approx(1.0 - 0.42 / 0.63)


class TestComputeSkillScores:
    def test_skill_scores_table(self) -> None:
        df = pd.DataFrame(
            {
                "model_name": [
                    "climatology",
                    "persistence",
                    "random_forest",
                    "xgboost",
                    "climatology",
                    "persistence",
                    "random_forest",
                ],
                "split_protocol": [
                    "random", "random", "random", "random",
                    "spatial_block", "spatial_block", "spatial_block",
                ],
                "rmse": [
                    0.80, 0.72, 0.55, 0.50,
                    0.85, 0.78, 0.62,
                ],
            }
        )
        result = compute_skill_scores(
            df, baseline_names=["climatology", "persistence"]
        )

        # random_forest vs climatology on random split: 1 - 0.55/0.80 = 0.3125
        rf_random_clim = result[
            (result["model_name"] == "random_forest")
            & (result["split_protocol"] == "random")
            & (result["baseline"] == "climatology")
        ]
        assert len(rf_random_clim) == 1
        assert rf_random_clim.iloc[0]["skill_score"] == pytest.approx(1.0 - 0.55 / 0.80)

    def test_missing_baseline_yields_nan(self) -> None:
        df = pd.DataFrame(
            {
                "model_name": ["random_forest", "xgboost"],
                "split_protocol": ["random", "random"],
                "rmse": [0.55, 0.50],
            }
        )
        result = compute_skill_scores(
            df, baseline_names=["climatology", "persistence"]
        )
        # No baselines → all skill scores should be NaN
        assert result["skill_score"].isna().all()

    def test_no_baseline_rows(self) -> None:
        """When there are no baseline rows, all skill scores are NaN with notes."""
        df = pd.DataFrame(
            {
                "model_name": ["rf", "xgb"],
                "split_protocol": ["random", "random"],
                "rmse": [0.55, 0.50],
            }
        )
        result = compute_skill_scores(df, baseline_names=["climatology"])
        assert (result["skill_score"].isna()).all()
        assert (result["note"].notna()).all()

    def test_same_model_and_baseline_skipped(self) -> None:
        """Baselines should not have skill scores computed against themselves."""
        df = pd.DataFrame(
            {
                "model_name": ["climatology", "persistence", "rf"],
                "split_protocol": ["random", "random", "random"],
                "rmse": [0.80, 0.72, 0.55],
            }
        )
        result = compute_skill_scores(df, baseline_names=["climatology", "persistence"])
        # Only "rf" should have skill scores; baselines are excluded.
        assert set(result["model_name"].unique()) == {"rf"}

    def test_negative_skill_score(self) -> None:
        """Model worse than baseline → negative skill."""
        df = pd.DataFrame(
            {
                "model_name": ["climatology", "bad_model"],
                "split_protocol": ["random", "random"],
                "rmse": [0.50, 0.90],
            }
        )
        result = compute_skill_scores(df, baseline_names=["climatology"])
        ss = result.iloc[0]["skill_score"]
        assert ss < 0


# ---------------------------------------------------------------------------
#  OOD degradation
# ---------------------------------------------------------------------------


class TestOODDegradation:
    def test_no_degradation(self) -> None:
        assert compute_ood_degradation(rmse_ood=0.5, rmse_random=0.5) == 0.0

    def test_positive_degradation(self) -> None:
        # RMSE goes from 0.5 → 0.75 → 50 % worse
        assert compute_ood_degradation(0.75, 0.5) == 0.5

    def test_negative_degradation(self) -> None:
        # OOD is actually better — unusual but possible
        assert compute_ood_degradation(0.3, 0.5) == -0.4

    def test_random_zero_returns_nan(self) -> None:
        assert np.isnan(compute_ood_degradation(0.5, 0.0))

    def test_negative_rmse_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_ood_degradation(-0.5, 1.0)
        with pytest.raises(ValueError):
            compute_ood_degradation(0.5, -1.0)


class TestOODDegradationTable:
    def test_basic_table(self) -> None:
        df = pd.DataFrame(
            {
                "model_name": [
                    "rf", "rf", "rf", "rf",
                    "xgb", "xgb", "xgb",
                ],
                "split_protocol": [
                    "random", "spatial_block", "temporal", "region_transfer",
                    "random", "spatial_block", "temporal",
                ],
                "rmse": [
                    0.50, 0.62, 0.58, 0.70,
                    0.48, 0.60, 0.55,
                ],
            }
        )
        result = compute_ood_degradation_table(df, reference_split="random")

        # rf spatial_block: (0.62 - 0.50) / 0.50 = 0.24
        rf_sp = result[
            (result["model_name"] == "rf")
            & (result["ood_split"] == "spatial_block")
        ]
        assert rf_sp.iloc[0]["ood_degradation"] == pytest.approx(0.24)

        # xgb temporal: (0.55 - 0.48) / 0.48 ≈ 0.1458
        xgb_t = result[
            (result["model_name"] == "xgb")
            & (result["ood_split"] == "temporal")
        ]
        assert xgb_t.iloc[0]["ood_degradation"] == pytest.approx(
            (0.55 - 0.48) / 0.48
        )

    def test_missing_random_reference(self) -> None:
        df = pd.DataFrame(
            {
                "model_name": ["rf", "rf"],
                "split_protocol": ["spatial_block", "temporal"],
                "rmse": [0.62, 0.58],
            }
        )
        result = compute_ood_degradation_table(df)
        assert result["ood_degradation"].isna().all()
        assert (result["note"].notna()).all()

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing required"):
            compute_ood_degradation_table(df)
        with pytest.raises(ValueError, match="missing required"):
            compute_skill_scores(df, baseline_names=["climatology"])
