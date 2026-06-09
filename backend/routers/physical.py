"""Physical consistency audit endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.services.physical_service import (
    get_physical_summary,
    get_regional_sensitivity,
)

router = APIRouter(tags=["physical"])


@router.get("/physical-consistency/summary")
def physical_summary():
    """Return the physical consistency audit summary."""
    return get_physical_summary()


@router.get("/physical-consistency/regional-sensitivity")
def regional_sensitivity(
    feature: str | None = Query(None),
    region: str | None = Query(None),
):
    """Return regional feature sensitivity data."""
    return get_regional_sensitivity(feature=feature, region=region)
