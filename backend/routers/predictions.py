"""Prediction and residual endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.config import DEFAULT_PREDICTION_LIMIT, get_experiment_dir
from backend.schemas import PredictionRecord, PredictionSummary, ResidualRecord
from backend.services.prediction_service import (
    get_prediction_summary,
    get_predictions,
    get_residuals,
)

router = APIRouter(tags=["predictions"])


@router.get("/experiments/{experiment_id}/predictions", response_model=list[PredictionRecord])
def experiment_predictions(
    experiment_id: str,
    model: str | None = Query(None, description="Filter by model name"),
    validation_strategy: str | None = Query(None, description="Filter by validation strategy"),
    region: str | None = Query(None, description="Filter by region"),
    feature_set: str | None = Query(None, description="Filter by feature set"),
    year: int | None = Query(None, description="Filter by year"),
    month: int | None = Query(None, ge=1, le=12, description="Filter by month"),
    limit: int = Query(DEFAULT_PREDICTION_LIMIT, ge=1, le=5000, description="Max rows to return"),
) -> list[PredictionRecord]:
    """Return prediction rows for an experiment."""
    exp_dir = get_experiment_dir(experiment_id)
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    predictions = get_predictions(
        experiment_id, model=model, validation_strategy=validation_strategy,
        region=region, feature_set=feature_set, year=year, month=month, limit=limit,
    )
    return [PredictionRecord(**p) for p in predictions]


@router.get("/experiments/{experiment_id}/residuals", response_model=list[ResidualRecord])
def experiment_residuals(
    experiment_id: str,
    region: str | None = Query(None, description="Filter by region"),
    limit: int = Query(DEFAULT_PREDICTION_LIMIT, ge=1, le=5000),
) -> list[ResidualRecord]:
    """Return residuals (actual - prediction) for an experiment."""
    exp_dir = get_experiment_dir(experiment_id)
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    residuals = get_residuals(experiment_id, region=region, limit=limit)
    return [ResidualRecord(**r) for r in residuals]


@router.get("/experiments/{experiment_id}/prediction-summary", response_model=PredictionSummary)
def prediction_summary(
    experiment_id: str,
    region: str | None = Query(None),
) -> PredictionSummary:
    """Return summary statistics for prediction residuals."""
    exp_dir = get_experiment_dir(experiment_id)
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    summary = get_prediction_summary(experiment_id, region=region)
    return PredictionSummary(**summary)
