"""Seed helpers for local backend platform demos."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import models as db_models
from backend.database import get_db, get_engine


def seed_platform_defaults(db: Session) -> dict[str, int]:
    """Create the minimal platform rows needed for a local submission demo."""
    dataset = db.scalar(select(db_models.Dataset).where(db_models.Dataset.name == "climatenet-demo"))
    if dataset is None:
        dataset = db_models.Dataset(
            name="climatenet-demo",
            version="v1",
            description="Default ClimateNet-Bench demo dataset.",
            storage_uri="local://demo",
            metadata_json={"source": "seed"},
        )
        db.add(dataset)
        db.flush()

    task = db.scalar(
        select(db_models.BenchmarkTask).where(
            db_models.BenchmarkTask.dataset_id == dataset.id,
            db_models.BenchmarkTask.name == "evaporation-anomaly",
        )
    )
    if task is None:
        task = db_models.BenchmarkTask(
            dataset_id=dataset.id,
            name="evaporation-anomaly",
            target_variable="evaporation_anomaly",
            task_type="regression",
            description="Predict land evaporation anomaly.",
        )
        db.add(task)
        db.flush()

    split = db.scalar(
        select(db_models.SplitProtocol).where(
            db_models.SplitProtocol.benchmark_task_id == task.id,
            db_models.SplitProtocol.name == "random-v1",
        )
    )
    if split is None:
        split = db_models.SplitProtocol(
            dataset_id=dataset.id,
            benchmark_task_id=task.id,
            name="random-v1",
            protocol_type="random",
            config={"seed": 42, "test_size": 0.2},
        )
        db.add(split)
        db.flush()

    model = db.scalar(select(db_models.BenchmarkModel).where(db_models.BenchmarkModel.name == "demo-baseline"))
    if model is None:
        model = db_models.BenchmarkModel(
            name="demo-baseline",
            version="v1",
            model_type="baseline",
            description="Default model registration for local submission demos.",
            metadata_json={"source": "seed"},
        )
        db.add(model)
        db.flush()

    db.commit()
    return {
        "dataset_id": dataset.id,
        "benchmark_task_id": task.id,
        "split_protocol_id": split.id,
        "model_id": model.id,
    }


def seed_configured_database() -> dict[str, int]:
    """Seed the configured DATABASE_URL database."""
    engine = get_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is required to seed the backend platform database.")

    db_iter = get_db()
    db = next(db_iter)
    try:
        return seed_platform_defaults(db)
    finally:
        try:
            next(db_iter)
        except StopIteration:
            pass


def main() -> int:
    try:
        ids = seed_configured_database()
    except RuntimeError as exc:
        print(str(exc))
        return 1

    print(
        "Seeded platform defaults: "
        f"dataset_id={ids['dataset_id']}, "
        f"benchmark_task_id={ids['benchmark_task_id']}, "
        f"split_protocol_id={ids['split_protocol_id']}, "
        f"model_id={ids['model_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
