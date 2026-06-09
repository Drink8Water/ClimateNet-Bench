"""ClimateNet-Bench evaluation metrics, skill scores, OOD analysis, and uncertainty.

This subpackage provides:

- **Primary metrics**: MAE, RMSE, R²
- **Skill scores**: improvement over climatology and persistence baselines
- **OOD degradation**: performance drop under distribution shift
- **Conformal prediction**: split-conformal prediction intervals
- **Calibration**: coverage and interval-width reports
- **Physical consistency**: model behaviour audit against atmospheric-science expectations
"""

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
from climatenet.evaluation.metrics import evaluate_regression, mae, r2, rmse
from climatenet.evaluation.ood_degradation import compute_ood_degradation, compute_ood_degradation_table
from climatenet.evaluation.physical_consistency import (
    AUDIT_FEATURES,
    check_monotonic_trend,
    compute_feature_sensitivity,
    compute_regional_sensitivity,
    is_consistent_with_expectation,
    run_physical_audit,
)
from climatenet.evaluation.skill_score import compute_skill_scores, skill_score
