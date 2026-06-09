"""Uncertainty and calibration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.services.uncertainty_service import (
    get_calibration,
    get_experiment_intervals,
)

router = APIRouter(tags=["uncertainty"])


@router.get("/uncertainty/calibration")
def calibration(
    model_name: str | None = Query(None),
    split_protocol: str | None = Query(None),
):
    """Return uncertainty calibration table."""
    return get_calibration(model_name=model_name, split_protocol=split_protocol)


@router.get("/experiments/{experiment_id}/intervals")
def experiment_intervals(
    experiment_id: str,
    limit: int = Query(500, ge=1, le=5000),
    region: str | None = Query(None),
):
    """Return prediction intervals for one experiment."""
    intervals = get_experiment_intervals(experiment_id, limit=limit, region=region)
    if not intervals:
        raise HTTPException(
            status_code=404,
            detail=f"No intervals found for experiment '{experiment_id}'",
        )
    return intervals
