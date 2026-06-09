"""Shared configuration for the climate ML pipeline."""

from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

SAMPLE_DATA_PATH = PROCESSED_DATA_DIR / "sample_climate_data.csv"
ERA5_DATA_PATH = PROCESSED_DATA_DIR / "era5_climate_data.csv"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
RANDOM_SPLIT_METRICS_PATH = OUTPUTS_DIR / "random_split_metrics.json"
SPATIAL_HOLDOUT_METRICS_PATH = OUTPUTS_DIR / "spatial_holdout_metrics.json"
REGION_TRANSFER_METRICS_PATH = OUTPUTS_DIR / "region_transfer_metrics.json"
ALL_METRICS_PATH = OUTPUTS_DIR / "all_metrics.csv"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_BY_REGION_PATH = OUTPUTS_DIR / "feature_importance_by_region.csv"
PREDICTION_PLOT_PATH = OUTPUTS_DIR / "prediction_vs_actual.png"
FEATURE_IMPORTANCE_PLOT_PATH = OUTPUTS_DIR / "feature_importance.png"
METRICS_PLOT_PATH = OUTPUTS_DIR / "validation_metrics.png"
SHAP_SUMMARY_PATH = OUTPUTS_DIR / "shap_summary.png"
REGIONAL_SHAP_SAHARA_PATH = OUTPUTS_DIR / "regional_shap_sahara.png"
REGIONAL_SHAP_EAST_CHINA_PATH = OUTPUTS_DIR / "regional_shap_east_china.png"

REGIONS = {
    "Sahara": {
        "lat_range": (18.0, 30.0),
        "lon_range": (-15.0, 30.0),
    },
    "East China": {
        "lat_range": (24.0, 36.0),
        "lon_range": (110.0, 122.0),
    },
}

ERA5_REGIONS = {
    "Sahara": {
        "lat_range": (15.0, 30.0),
        "lon_range": (-20.0, 30.0),
        "cds_area": [30.0, -20.0, 15.0, 30.0],  # north, west, south, east
    },
    "East China": {
        "lat_range": (20.0, 35.0),
        "lon_range": (105.0, 122.0),
        "cds_area": [35.0, 105.0, 20.0, 122.0],  # north, west, south, east
    },
}

ERA5_YEARS = range(2019, 2024)
ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "total_evaporation",
    "volumetric_soil_water_layer_1",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

ERA5_RAW_DIR = RAW_DATA_DIR / "era5_land"

YEARS = range(2010, 2021)
GRID_POINTS_PER_REGION = 30

TARGET_COLUMN = "evaporation_anomaly"

MODEL_FEATURES = [
    "latitude",
    "longitude",
    "temperature",
    "precipitation",
    "radiation",
    "soil_moisture",
    "u_wind",
    "v_wind",
    "wind_speed",
    "month_sin",
    "month_cos",
    "temperature_anomaly",
    "precipitation_anomaly",
    "radiation_anomaly",
    "soil_moisture_anomaly",
    "dryness_proxy",
    "saturation_vapor_pressure",
]


def ensure_directories() -> None:
    """Create project data/output directories if they do not already exist."""
    for directory in [RAW_DATA_DIR, ERA5_RAW_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
