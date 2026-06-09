"""Pydantic response models for the ClimateNet API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# ── Health ──────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    database: str = "file-based"


# ── Summary ─────────────────────────────────────────────────────────
class PipelineStep(BaseModel):
    step: int
    label: str
    description: str


class ArchitectureLayer(BaseModel):
    layer: str
    technologies: list[str]
    description: str


class ProjectSummaryResponse(BaseModel):
    title: str
    subtitle: str
    description: str
    pipeline_steps: list[PipelineStep]
    architecture_layers: list[ArchitectureLayer]


class DatasetSummaryResponse(BaseModel):
    data_source: str
    regions: list[str]
    target_variable: str
    models: list[str]
    total_records: int
    timespan: str


# ── Experiments ─────────────────────────────────────────────────────
class ExperimentSummary(BaseModel):
    experiment_id: str
    model_name: str
    validation_strategy: str
    feature_set: str
    train_region: str
    test_region: str
    train_period: str
    test_period: str
    n_train: int | None = None
    n_test: int | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None


class ExperimentDetail(BaseModel):
    experiment_id: str
    config: dict[str, Any]
    metrics: list[dict[str, Any]]
    metrics_count: int
    prediction_count: int
    feature_count: int


class ExperimentSummaryResponse(BaseModel):
    total_experiments: int
    best_r2: float | None = None
    best_rmse: float | None = None
    best_model: str | None = None
    regions: list[str]
    models: list[str]
    strategies: list[str]
    feature_sets: list[str]


# ── Metrics ─────────────────────────────────────────────────────────
class MetricRecord(BaseModel):
    validation_strategy: str
    train_region: str | None = None
    test_region: str | None = None
    train_period: str | None = None
    test_period: str | None = None
    feature_set: str | None = None
    model: str
    model_name: str | None = None
    n_train: int | None = None
    n_test: int | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None


# ── Predictions ─────────────────────────────────────────────────────
class PredictionRecord(BaseModel):
    region: str | None = None
    year: int | None = None
    month: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    validation_strategy: str | None = None
    train_region: str | None = None
    test_region: str | None = None
    feature_set: str | None = None
    model: str | None = None
    actual: float | None = None
    prediction: float | None = None


class ResidualRecord(BaseModel):
    year: int | None = None
    month: int | None = None
    region: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    actual: float | None = None
    prediction: float | None = None
    residual: float | None = None


class PredictionSummary(BaseModel):
    mean_residual: float
    residual_std: float
    max_absolute_error: float
    prediction_count: int


# ── Feature Importance ──────────────────────────────────────────────
class FeatureImportanceRecord(BaseModel):
    validation_strategy: str | None = None
    train_region: str | None = None
    test_region: str | None = None
    model: str | None = None
    feature: str
    importance: float | None = None
    importance_type: str | None = None
    importance_mean: float | None = None
    importance_std: float | None = None
    region: str | None = None


class ShapResponse(BaseModel):
    available: bool
    experiment_id: str
    plot_paths: list[str]


# ── Spatial ─────────────────────────────────────────────────────────
class SpatialGridRecord(BaseModel):
    region: str
    latitude: float
    longitude: float
    year: int
    month: int
    value: float | None = None


class TimeSeriesRecord(BaseModel):
    region: str
    year: int
    month: int
    temperature: float | None = None
    precipitation: float | None = None
    radiation: float | None = None
    soil_moisture: float | None = None
    evaporation_anomaly: float | None = None


class GridCellDetail(BaseModel):
    region: str
    year: int
    month: int
    latitude: float
    longitude: float
    temperature: float | None = None
    precipitation: float | None = None
    radiation: float | None = None
    soil_moisture: float | None = None
    evaporation: float | None = None
    evaporation_anomaly: float | None = None
    wind_speed: float | None = None


# ── Ablation Study ──────────────────────────────────────────────────
class AblationStudyRecord(BaseModel):
    feature_set: str
    model_name: str
    validation_strategy: str
    mean_mae: float | None = None
    mean_rmse: float | None = None
    mean_r2: float | None = None
    best_rmse: float | None = None


# ── Legacy compatibility aliases ────────────────────────────────────
MetricResponse = MetricRecord
PredictionResponse = PredictionRecord
FeatureImportanceResponse = FeatureImportanceRecord
TimeSeriesResponse = TimeSeriesRecord
RegionalSummaryResponse = DatasetSummaryResponse
