"""Spatial grid and timeseries endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.config import ANOMALY_VARIABLES, REGIONS
from backend.schemas import GridCellDetail, SpatialGridRecord, TimeSeriesRecord
from backend.services.spatial_service import (
    get_grid_cell_detail,
    get_spatial_grid,
    get_timeseries,
)

router = APIRouter(tags=["spatial"])


@router.get("/spatial-grid", response_model=list[SpatialGridRecord])
def spatial_grid(
    variable: str = Query("evaporation_anomaly", description="Variable to display on the grid"),
    region: str | None = Query(None, description="Filter by region"),
    year: int | None = Query(None, description="Filter by year"),
    month: int | None = Query(None, ge=1, le=12, description="Filter by month"),
    limit: int = Query(2000, ge=1, le=5000),
) -> list[SpatialGridRecord]:
    """Return grid points with selected variable values for spatial visualization."""
    if variable not in ANOMALY_VARIABLES and variable not in [
        "temperature", "precipitation", "radiation", "soil_moisture", "wind_speed", "evaporation"
    ]:
        variable = "evaporation_anomaly"

    data = get_spatial_grid(variable=variable, region=region, year=year, month=month, limit=limit)
    return [SpatialGridRecord(**row) for row in data]


@router.get("/timeseries", response_model=list[TimeSeriesRecord])
def timeseries(
    region: str | None = Query(None, description="Filter by region"),
    variable: str = Query("evaporation_anomaly", description="Anomaly variable to track"),
) -> list[TimeSeriesRecord]:
    """Return monthly aggregated timeseries by region."""
    data = get_timeseries(region=region, variable=variable)
    return [TimeSeriesRecord(**row) for row in data]


@router.get("/grid-cell-detail", response_model=GridCellDetail)
def grid_cell_detail(
    latitude: float = Query(..., description="Grid cell latitude"),
    longitude: float = Query(..., description="Grid cell longitude"),
    year: int = Query(..., description="Year"),
    month: int = Query(..., ge=1, le=12, description="Month"),
) -> GridCellDetail:
    """Return full detail for a specific grid cell at a given time."""
    detail = get_grid_cell_detail(latitude=latitude, longitude=longitude, year=year, month=month)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for lat={latitude}, lon={longitude}, year={year}, month={month}",
        )
    return GridCellDetail(**detail)
