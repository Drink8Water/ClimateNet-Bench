"""Project and dataset summary endpoints (static + file-driven)."""

from __future__ import annotations

from fastapi import APIRouter

from backend.config import FEATURES_PATH, MODEL_NAMES, REGIONS
from backend.schemas import (
    ArchitectureLayer,
    DatasetSummaryResponse,
    PipelineStep,
    ProjectSummaryResponse,
)

router = APIRouter(tags=["summary"])


@router.get("/project-summary", response_model=ProjectSummaryResponse)
def project_summary() -> ProjectSummaryResponse:
    """Return project metadata and pipeline description."""
    return ProjectSummaryResponse(
        title="ClimateNet",
        subtitle="Domain-informed Spatio-temporal ML System for Land Evaporation Anomaly Modeling",
        description=(
            "ClimateNet is a reproducible climate data machine learning system that processes "
            "ERA5-Land NetCDF data, constructs physically informed spatio-temporal features, "
            "trains machine learning models to predict land evaporation anomalies, evaluates "
            "spatial generalization, and explains model behavior."
        ),
        pipeline_steps=[
            PipelineStep(step=1, label="ERA5-Land NetCDF", description="Retrieve gridded climate reanalysis data via CDS API"),
            PipelineStep(step=2, label="xarray Preprocessing", description="Read NetCDF, subset regions, compute monthly climatology and anomalies"),
            PipelineStep(step=3, label="Feature Engineering", description="Construct physically informed predictors: wind speed, dryness proxy, saturation vapor pressure, cyclic month encoding"),
            PipelineStep(step=4, label="ML Experiments", description="Train Linear Regression, Random Forest, XGBoost, LightGBM, and TCN models across 4 validation strategies and 4 feature sets"),
            PipelineStep(step=5, label="FastAPI Backend", description="Serve experiment results, predictions, feature importance, and spatial data via REST API"),
            PipelineStep(step=6, label="Vue Dashboard", description="Interactive visualization dashboard for experiment management and model analysis"),
        ],
        architecture_layers=[
            ArchitectureLayer(layer="Data Processing", technologies=["xarray", "NetCDF4", "pandas"], description="Gridded climate data ingestion and preprocessing"),
            ArchitectureLayer(layer="Feature Engineering", technologies=["NumPy", "SciPy"], description="Physical feature construction: wind speed, saturation vapor pressure, dryness proxy"),
            ArchitectureLayer(layer="Modeling", technologies=["scikit-learn", "XGBoost", "LightGBM", "PyTorch"], description="Tree-based and deep learning models for anomaly prediction"),
            ArchitectureLayer(layer="Experiment Management", technologies=["YAML configs", "CSV/JSON outputs"], description="Versioned experiments with reproducible config snapshots"),
            ArchitectureLayer(layer="Visualization", technologies=["FastAPI", "Vue 3", "ECharts", "Tailwind CSS"], description="REST API serving experiment data to an interactive dashboard"),
        ],
    )


@router.get("/dataset-summary", response_model=DatasetSummaryResponse)
def dataset_summary() -> DatasetSummaryResponse:
    """Return dataset metadata derived from the features file."""
    total_records = 0
    if FEATURES_PATH.exists():
        try:
            total_records = sum(1 for _ in open(FEATURES_PATH, encoding="utf-8")) - 1
        except Exception:
            pass

    return DatasetSummaryResponse(
        data_source="ERA5-Land (synthetic sample for MVP)",
        regions=REGIONS,
        target_variable="evaporation_anomaly",
        models=[m for m in MODEL_NAMES],
        total_records=total_records,
        timespan="2010–2020",
    )
