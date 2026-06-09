"""Experiment listing and detail endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.schemas import ExperimentDetail, ExperimentSummary, ExperimentSummaryResponse
from backend.services.experiment_service import (
    get_all_experiments,
    get_experiment_detail,
    get_experiment_summary,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentSummary])
def list_experiments(
    model_name: str | None = Query(None, description="Filter by model name"),
    validation_strategy: str | None = Query(None, description="Filter by validation strategy"),
    feature_set: str | None = Query(None, description="Filter by feature set"),
    region: str | None = Query(None, description="Filter by region"),
) -> list[ExperimentSummary]:
    """List all experiments with optional filters."""
    experiments = get_all_experiments(
        model_name=model_name,
        validation_strategy=validation_strategy,
        feature_set=feature_set,
        region=region,
    )
    return [ExperimentSummary(**exp) for exp in experiments]


@router.get("/summary", response_model=ExperimentSummaryResponse)
def experiment_summary() -> ExperimentSummaryResponse:
    """Return aggregated experiment KPIs."""
    summary = get_experiment_summary()
    return ExperimentSummaryResponse(**summary)


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def experiment_detail(experiment_id: str) -> ExperimentDetail:
    """Return configuration and metrics for a single experiment."""
    detail = get_experiment_detail(experiment_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return ExperimentDetail(**detail)
