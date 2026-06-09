"""ClimateNet FastAPI application.

Serves ML experiment outputs from local CSV/JSON files (no database required).
Legacy PostgreSQL endpoints remain available when DATABASE_URL is configured.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import (
    attribution,
    benchmark,
    comparison,
    experiments,
    leaderboard,
    physical,
    predictions,
    spatial,
    summary,
    uncertainty,
)
from backend.schemas import HealthResponse

app = FastAPI(
    title="ClimateNet-Bench API",
    description="REST API for the ClimateNet-Bench benchmark: leaderboard, experiments, predictions, uncertainty, and physical consistency audit.",
    version="0.3.0",
)

# Allow the Vue dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Benchmark routers (new) ───────────────────────────────────────
app.include_router(benchmark.router)
app.include_router(leaderboard.router)
app.include_router(uncertainty.router)
app.include_router(physical.router)

# ── Existing file-based routers ────────────────────────────────────
app.include_router(summary.router)
app.include_router(experiments.router)
app.include_router(comparison.router)
app.include_router(predictions.router)
app.include_router(attribution.router)
app.include_router(spatial.router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — always available (file-based, no database required)."""
    return HealthResponse(status="ok", database="file-based")


# ── Legacy PostgreSQL endpoints (only active when DATABASE_URL is set) ──
try:
    from backend.database import get_db  # noqa: F401

    _db_available = True
except RuntimeError:
    _db_available = False

if _db_available:
    from backend import crud, schemas as legacy_schemas
    from backend.database import get_db
    from fastapi import Depends, Query
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    @app.get("/db/health", response_model=legacy_schemas.HealthResponse)
    def db_health(db: Session = Depends(get_db)) -> legacy_schemas.HealthResponse:
        """Check API and database connectivity."""
        db.execute(text("SELECT 1"))
        return legacy_schemas.HealthResponse(status="ok", database="connected")

    @app.get("/db/metrics", response_model=list[legacy_schemas.MetricResponse])
    def db_metrics(
        model: str | None = None,
        validation_strategy: str | None = None,
        db: Session = Depends(get_db),
    ) -> list[dict[str, object]]:
        """Return model metrics from PostgreSQL."""
        return crud.get_metrics(db, model=model, validation_strategy=validation_strategy)

    @app.get("/db/feature-importance", response_model=list[legacy_schemas.FeatureImportanceResponse])
    def db_feature_importance(
        region: str | None = None,
        model: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        db: Session = Depends(get_db),
    ) -> list[dict[str, object]]:
        """Return feature importance from PostgreSQL."""
        return crud.get_feature_importance(db, region=region, model=model, limit=limit)

    @app.get("/db/timeseries", response_model=list[legacy_schemas.TimeSeriesResponse])
    def db_timeseries(region: str | None = None, db: Session = Depends(get_db)) -> list[dict[str, object]]:
        """Return monthly regional mean climate features from PostgreSQL."""
        return crud.get_timeseries(db, region=region)

    @app.get("/db/predictions", response_model=list[legacy_schemas.PredictionResponse])
    def db_predictions(
        model: str | None = None,
        validation_strategy: str | None = None,
        region: str | None = None,
        limit: int = Query(default=500, ge=1, le=5000),
        db: Session = Depends(get_db),
    ) -> list[dict[str, object]]:
        """Return saved prediction rows from PostgreSQL."""
        return crud.get_predictions(
            db, model=model, validation_strategy=validation_strategy,
            region=region, limit=limit,
        )

    @app.get("/db/regional-summary", response_model=list[legacy_schemas.RegionalSummaryResponse])
    def db_regional_summary(db: Session = Depends(get_db)) -> list[dict[str, object]]:
        """Return regional climate summary from PostgreSQL."""
        return crud.get_regional_summary(db)
