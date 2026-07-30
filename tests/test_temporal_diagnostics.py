"""Unit tests for temporal failure diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climatenet.diagnostics.temporal_failure import (
    compute_error_metrics,
    summarize_feature_shift,
    validate_repeated_spatial_plan,
)


def test_error_metrics_include_bias_tail_and_direction() -> None:
    frame = pd.DataFrame(
        {"y_true": [0.0, 1.0, 2.0], "y_pred": [1.0, 1.0, 1.0]}
    )

    metrics = compute_error_metrics(frame)

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["bias"] == pytest.approx(0)
    assert metrics["overprediction_ratio"] == pytest.approx(1 / 3)
    assert metrics["underprediction_ratio"] == pytest.approx(1 / 3)
    assert metrics["residual_p99"] >= metrics["residual_p95"]
    assert metrics["residual_p95"] >= metrics["residual_p90"]
    assert metrics["abs_residual_p90"] >= metrics["abs_residual_p50"]


def test_feature_shift_reports_smd_quantiles_and_range_coverage() -> None:
    result = summarize_feature_shift([0, 1, 2, 3], [2, 3, 4, 5])

    assert result["standardized_mean_difference"] > 0
    assert result["test_p50"] > result["train_p50"]
    assert result["test_outside_train_range_ratio"] == pytest.approx(0.5)


def test_repeated_spatial_plan_guardrails() -> None:
    valid = {
        "folds": 5,
        "region_stratified": True,
        "no_grid_cell_overlap": True,
        "models": ["linear_regression", "lightgbm"],
        "feature_sets": ["full"],
    }
    validate_repeated_spatial_plan(valid)

    invalid = dict(valid, region_stratified=False)
    with pytest.raises(ValueError, match="stratified"):
        validate_repeated_spatial_plan(invalid)
