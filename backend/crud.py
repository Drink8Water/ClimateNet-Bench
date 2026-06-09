"""Read-only database queries used by the FastAPI backend."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


def fetch_all(db: Session, query: str, params: dict[str, object] | None = None) -> list[dict[str, object]]:
    """Execute a SELECT query and return dictionaries."""
    result = db.execute(text(query), params or {})
    return [dict(row._mapping) for row in result]


def get_metrics(db: Session, model: str | None = None, validation_strategy: str | None = None) -> list[dict[str, object]]:
    """Return model metrics with optional filters."""
    query = """
        SELECT validation_strategy, train_region, test_region, model, n_train, n_test, mae, rmse, r2
        FROM model_metrics
        WHERE (:model IS NULL OR model = :model)
          AND (:validation_strategy IS NULL OR validation_strategy = :validation_strategy)
        ORDER BY validation_strategy, train_region, test_region, rmse
    """
    return fetch_all(db, query, {"model": model, "validation_strategy": validation_strategy})


def get_feature_importance(db: Session, region: str | None = None, model: str | None = None, limit: int = 50) -> list[dict[str, object]]:
    """Return feature importance rows."""
    query = """
        SELECT region, validation_strategy, model, feature, importance, importance_mean, importance_std
        FROM feature_importance
        WHERE (:region IS NULL OR region = :region)
          AND (:model IS NULL OR model = :model)
        ORDER BY COALESCE(importance_mean, importance, 0) DESC
        LIMIT :limit
    """
    return fetch_all(db, query, {"region": region, "model": model, "limit": limit})


def get_timeseries(db: Session, region: str | None = None) -> list[dict[str, object]]:
    """Return monthly regional averages from the feature table."""
    query = """
        SELECT
            region,
            year,
            month,
            AVG(temperature) AS temperature,
            AVG(precipitation) AS precipitation,
            AVG(radiation) AS radiation,
            AVG(soil_moisture) AS soil_moisture,
            AVG(evaporation_anomaly) AS evaporation_anomaly
        FROM climate_features
        WHERE (:region IS NULL OR region = :region)
        GROUP BY region, year, month
        ORDER BY region, year, month
    """
    return fetch_all(db, query, {"region": region})


def get_predictions(
    db: Session,
    model: str | None = None,
    validation_strategy: str | None = None,
    region: str | None = None,
    limit: int = 500,
) -> list[dict[str, object]]:
    """Return saved model predictions."""
    query = """
        SELECT region, year, month, latitude, longitude, validation_strategy,
               train_region, test_region, model, actual, prediction
        FROM model_predictions
        WHERE (:model IS NULL OR model = :model)
          AND (:validation_strategy IS NULL OR validation_strategy = :validation_strategy)
          AND (:region IS NULL OR region = :region)
        ORDER BY year, month, region
        LIMIT :limit
    """
    return fetch_all(
        db,
        query,
        {"model": model, "validation_strategy": validation_strategy, "region": region, "limit": limit},
    )


def get_regional_summary(db: Session) -> list[dict[str, object]]:
    """Return high-level regional climate summary statistics."""
    query = """
        SELECT
            region,
            COUNT(*) AS n_records,
            AVG(temperature) AS mean_temperature,
            AVG(precipitation) AS mean_precipitation,
            AVG(radiation) AS mean_radiation,
            AVG(soil_moisture) AS mean_soil_moisture,
            AVG(evaporation_anomaly) AS mean_evaporation_anomaly
        FROM climate_features
        GROUP BY region
        ORDER BY region
    """
    return fetch_all(db, query)
