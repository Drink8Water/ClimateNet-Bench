"""Model comparison and ablation study endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.schemas import AblationStudyRecord
from backend.services.comparison_service import get_ablation_study, get_model_comparison

router = APIRouter(tags=["comparison"])


@router.get("/model-comparison")
def model_comparison(
    metric: str = Query("rmse", description="Metric to compare: r2, rmse, or mae"),
    validation_strategy: str | None = Query(None, description="Filter by validation strategy"),
    feature_set: str | None = Query(None, description="Filter by feature set"),
) -> list[dict]:
    """Return metric values grouped by model for comparison charts."""
    return get_model_comparison(
        metric=metric,
        validation_strategy=validation_strategy,
        feature_set=feature_set,
    )


@router.get("/ablation-study", response_model=list[AblationStudyRecord])
def ablation_study(
    experiment_id: str = Query("latest", description="Experiment ID"),
) -> list[AblationStudyRecord]:
    """Return ablation study comparing feature sets."""
    data = get_ablation_study(experiment_id)
    return [AblationStudyRecord(**row) for row in data]
