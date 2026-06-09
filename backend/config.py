"""Shared paths and constants for the ClimateNet backend."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
ALL_EXPERIMENTS_PATH = OUTPUTS_DIR / "all_experiments.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = DATA_PROCESSED_DIR / "features.csv"
LEGACY_OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

# ── Benchmark paths ──────────────────────────────────────────────
BENCHMARK_DIR = OUTPUTS_DIR / "benchmark"
BENCHMARK_EXPERIMENTS_DIR = BENCHMARK_DIR / "experiments"
BENCHMARK_SPLITS_DIR = BENCHMARK_DIR / "splits"

# Regions and variables available in the dataset
REGIONS = ["Sahara", "East China", "Amazon", "Central Europe", "Western US"]
ANOMALY_VARIABLES = [
    "evaporation_anomaly",
    "temperature_anomaly",
    "precipitation_anomaly",
    "radiation_anomaly",
    "soil_moisture_anomaly",
]
ALL_VARIABLES = ANOMALY_VARIABLES + [
    "temperature",
    "precipitation",
    "radiation",
    "soil_moisture",
    "wind_speed",
    "evaporation",
]

MODEL_NAMES = ["linear_regression", "random_forest", "xgboost", "lightgbm", "tcn"]
VALIDATION_STRATEGIES = ["random_split", "spatial_holdout", "temporal_holdout", "region_transfer"]
FEATURE_SETS = ["base", "meteorological", "physical", "full", "sequence"]

DEFAULT_PREDICTION_LIMIT = 500
MAX_PREDICTION_LIMIT = 5000


def get_experiment_dir(experiment_id: str) -> Path:
    """Return the directory for a given experiment ID."""
    return EXPERIMENTS_DIR / experiment_id


def list_experiment_ids() -> list[str]:
    """List all experiment directories, excluding 'latest' if it's a symlink."""
    if not EXPERIMENTS_DIR.exists():
        return []
    ids = []
    for entry in sorted(EXPERIMENTS_DIR.iterdir()):
        if entry.is_dir() and entry.name not in ("__pycache__",):
            ids.append(entry.name)
    return ids
