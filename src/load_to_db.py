"""Load Phase 3 CSV outputs into PostgreSQL tables."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import (
    ALL_METRICS_PATH,
    FEATURE_IMPORTANCE_BY_REGION_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURES_PATH,
    PREDICTIONS_PATH,
    PROJECT_ROOT,
)

SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "latest"


def get_database_url() -> str:
    """Read DATABASE_URL from environment variables or .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and edit the PostgreSQL URL."
        )
    return database_url


def require_file(path: Path, hint: str) -> None:
    """Raise a clear error if an expected CSV file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")


def experiment_dir() -> Path:
    """Return the experiment output directory used for DB loading."""
    configured = os.getenv("EXPERIMENT_OUTPUT_DIR")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_EXPERIMENT_DIR


def first_existing(paths: list[Path], hint: str) -> Path:
    """Return the first existing path or raise a clear error."""
    for path in paths:
        if path.exists():
            return path
    checked = "\n  ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the expected files exist:\n  {checked}\n{hint}")


def create_tables(engine: Engine) -> None:
    """Create database tables from sql/schema.sql."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing schema file: {SCHEMA_PATH}")

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    # Remove SQL line comments (--) so stray semicolons inside comments
    # don't produce empty fragments, then split on the real statement delimiters.
    clean_sql = re.sub(r"^\s*--.*$", "", schema_sql, flags=re.MULTILINE)
    statements = [s.strip() for s in clean_sql.split(";") if s.strip()]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def clear_tables(engine: Engine) -> None:
    """Clear project tables before loading a reproducible snapshot."""
    tables = ["model_predictions", "feature_importance", "model_metrics", "climate_features"]
    with engine.begin() as connection:
        for table in tables:
            connection.execute(text(f"DELETE FROM {table}"))


def load_dataframe(engine: Engine, data: pd.DataFrame, table_name: str) -> None:
    """Append a dataframe to an existing SQL table."""
    data.to_sql(table_name, engine, if_exists="append", index=False, method="multi", chunksize=1000)
    print(f"Loaded {len(data):,} rows into {table_name}")


def load_climate_features(engine: Engine) -> None:
    """Load engineered feature rows."""
    path = first_existing(
        [experiment_dir() / "features.csv", FEATURES_PATH],
        "Run: python scripts/run_pipeline.py and python scripts/run_experiment.py",
    )
    features = pd.read_csv(path)
    load_dataframe(engine, features, "climate_features")


def load_predictions(engine: Engine) -> None:
    """Load model prediction rows."""
    path = first_existing(
        [experiment_dir() / "predictions.csv", PREDICTIONS_PATH],
        "Run: python scripts/run_experiment.py",
    )
    predictions = pd.read_csv(path)
    load_dataframe(engine, predictions, "model_predictions")


def load_metrics(engine: Engine) -> None:
    """Load model metrics with database-friendly lowercase metric columns."""
    path = first_existing(
        [experiment_dir() / "metrics.csv", ALL_METRICS_PATH],
        "Run: python scripts/run_experiment.py",
    )
    metrics = pd.read_csv(path).rename(columns={"MAE": "mae", "RMSE": "rmse", "R2": "r2"})
    load_dataframe(engine, metrics, "model_metrics")


def load_feature_importance(engine: Engine) -> None:
    """Load global and regional feature importance into one table."""
    frames = []
    latest_importance = experiment_dir() / "feature_importance.csv"

    if latest_importance.exists():
        frames.append(pd.read_csv(latest_importance))
    elif FEATURE_IMPORTANCE_PATH.exists():
        frames.append(pd.read_csv(FEATURE_IMPORTANCE_PATH))

    if FEATURE_IMPORTANCE_BY_REGION_PATH.exists():
        regional = pd.read_csv(FEATURE_IMPORTANCE_BY_REGION_PATH)
        regional["validation_strategy"] = "random_split"
        regional["train_region"] = "mixed"
        regional["test_region"] = "mixed"
        regional["model"] = "best_tree_model"
        regional["importance"] = pd.NA
        regional["importance_type"] = "regional_permutation"
        frames.append(regional)

    if not frames:
        raise FileNotFoundError("Missing feature importance files. Run: python src/train.py and python src/explain.py")

    expected_columns = [
        "region",
        "validation_strategy",
        "train_region",
        "test_region",
        "model",
        "feature",
        "importance",
        "importance_type",
        "importance_mean",
        "importance_std",
    ]
    importance = pd.concat(frames, ignore_index=True)
    for column in expected_columns:
        if column not in importance.columns:
            importance[column] = pd.NA

    load_dataframe(engine, importance[expected_columns], "feature_importance")


def main() -> None:
    """Load all project outputs into PostgreSQL."""
    try:
        database_url = get_database_url()
    except RuntimeError as exc:
        sys.exit(str(exc))

    engine = create_engine(database_url)

    try:
        create_tables(engine)
        clear_tables(engine)
        load_climate_features(engine)
        load_predictions(engine)
        load_metrics(engine)
        load_feature_importance(engine)
    except SQLAlchemyError as exc:
        sys.exit(
            "Database operation failed. Check that PostgreSQL is running and DATABASE_URL is correct.\n"
            f"Details: {exc}"
        )

    print("Database load complete.")


if __name__ == "__main__":
    main()
