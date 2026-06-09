"""Feature attribution and explanation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.config import get_experiment_dir
from backend.schemas import FeatureImportanceRecord, ShapResponse
from backend.services.attribution_service import (
    get_feature_importance,
    get_local_explanations,
    get_shap_info,
)

router = APIRouter(tags=["attribution"])


@router.get("/experiments/{experiment_id}/feature-importance", response_model=list[FeatureImportanceRecord])
def feature_importance(
    experiment_id: str,
    model: str | None = Query(None, description="Filter by model"),
    validation_strategy: str | None = Query(None, description="Filter by validation strategy"),
    region: str | None = Query(None, description="Filter by region"),
    limit: int = Query(200, ge=1, le=1000),
) -> list[FeatureImportanceRecord]:
    """Return feature importance rows for an experiment."""
    exp_dir = get_experiment_dir(experiment_id)
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    data = get_feature_importance(
        experiment_id, model=model, validation_strategy=validation_strategy,
        region=region, limit=limit,
    )
    return [FeatureImportanceRecord(**row) for row in data]


@router.get("/experiments/{experiment_id}/shap", response_model=ShapResponse)
def shap_info(experiment_id: str) -> ShapResponse:
    """Return SHAP plot availability and paths for an experiment."""
    exp_dir = get_experiment_dir(experiment_id)
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    info = get_shap_info(experiment_id)
    return ShapResponse(**info)


@router.get("/experiments/{experiment_id}/local-explanations")
def local_explanations(
    experiment_id: str,
    model: str | None = Query(None),
    region: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    """Return top local explanation records."""
    exp_dir = get_experiment_dir(experiment_id)
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return get_local_explanations(experiment_id, model=model, region=region, limit=limit)
