"""Leaderboard, split-difficulty, and ablation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.services.leaderboard_service import (
    get_ablation_study,
    get_leaderboard,
    get_split_difficulty,
)

router = APIRouter(tags=["leaderboard"])


@router.get("/leaderboard")
def leaderboard(
    model_name: str | None = Query(None),
    split_protocol: str | None = Query(None),
    feature_set: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Return the ranked benchmark leaderboard."""
    return get_leaderboard(
        model_name=model_name,
        split_protocol=split_protocol,
        feature_set=feature_set,
        limit=limit,
    )


@router.get("/leaderboard/{split_protocol}")
def leaderboard_by_split(
    split_protocol: str,
    model_name: str | None = Query(None),
):
    """Return leaderboard filtered by split protocol."""
    return get_leaderboard(
        model_name=model_name,
        split_protocol=split_protocol,
    )


@router.get("/split-difficulty")
def split_difficulty():
    """Return split difficulty analysis (mean RMSE per protocol)."""
    return get_split_difficulty()


@router.get("/ablation-study")
def ablation_study(
    model_name: str | None = Query(None),
    split_protocol: str | None = Query(None),
):
    """Return ablation study results."""
    return get_ablation_study(
        model_name=model_name,
        split_protocol=split_protocol,
    )
