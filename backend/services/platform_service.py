"""Database-backed platform services for submissions and leaderboards."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import HTTPException, status
from starlette.datastructures import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend import models as db_models
from backend.artifacts import store_local_artifact
from backend.database import get_db, is_db_available
from backend.schemas import (
    EvaluationRunDetailResponse,
    EvaluationRunStatusResponse,
    SubmissionCreateRequest,
    SubmissionCreateResponse,
    SubmissionDetailResponse,
)
from backend.tasks import evaluate_submission_task


def db_backed_response(query_fn: Callable[[Session], Any], fallback: Any) -> Any:
    """Run a DB query when configured, otherwise return a mock response."""
    if not is_db_available():
        return fallback

    db_iter = get_db()
    db = next(db_iter)
    try:
        return query_fn(db)
    except SQLAlchemyError:
        return fallback
    finally:
        close_db_iter(db_iter)


def require_db() -> tuple[Generator[Any, None, None], Session]:
    """Return a DB session or raise an API-friendly 503."""
    if not is_db_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is required for this endpoint.",
        )

    db_iter = get_db()
    db = next(db_iter)
    return db_iter, db


def close_db_iter(db_iter: Generator[Any, None, None]) -> None:
    try:
        next(db_iter)
    except StopIteration:
        pass


def list_models() -> list[dict[str, object]]:
    def query(db: Session):
        rows = db.scalars(select(db_models.BenchmarkModel).order_by(db_models.BenchmarkModel.name)).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "version": row.version,
                "model_type": row.model_type,
                "description": row.description,
            }
            for row in rows
        ]

    return db_backed_response(query, [])


def list_datasets() -> list[dict[str, object]]:
    def query(db: Session):
        rows = db.scalars(select(db_models.Dataset).order_by(db_models.Dataset.name)).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "version": row.version,
                "description": row.description,
                "storage_uri": row.storage_uri,
            }
            for row in rows
        ]

    return db_backed_response(query, [])


def get_leaderboard(
    metric: str = "rmse",
    split_protocol: str | None = None,
    event_type: str | None = None,
    climate_zone: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    def query(db: Session):
        stmt = (
            select(
                db_models.EvaluationRun.id.label("evaluation_run_id"),
                db_models.Submission.id.label("submission_id"),
                db_models.BenchmarkModel.name.label("model_name"),
                db_models.SplitProtocol.name.label("split_protocol"),
                db_models.Metric.name.label("metric"),
                db_models.Metric.value.label("value"),
            )
            .join(db_models.Submission, db_models.EvaluationRun.submission_id == db_models.Submission.id)
            .join(db_models.BenchmarkModel, db_models.Submission.model_id == db_models.BenchmarkModel.id)
            .join(db_models.SplitProtocol, db_models.EvaluationRun.split_protocol_id == db_models.SplitProtocol.id)
            .join(db_models.Metric, db_models.Metric.evaluation_run_id == db_models.EvaluationRun.id)
            .where(db_models.EvaluationRun.status == "COMPLETED")
        )
        if split_protocol:
            stmt = stmt.where(db_models.SplitProtocol.name == split_protocol)
        ranked: dict[int, dict[str, object]] = {}
        for row in db.execute(stmt).all():
            record = dict(row._mapping)
            run_id = int(record["evaluation_run_id"])
            leaderboard_row = ranked.setdefault(
                run_id,
                {
                    "evaluation_run_id": run_id,
                    "submission_id": record["submission_id"],
                    "model_name": record["model_name"],
                    "split_protocol": record["split_protocol"],
                },
            )
            leaderboard_row[str(record["metric"])] = record["value"]

        if event_type:
            ranked = {
                run_id: row
                for run_id, row in ranked.items()
                if any(key.startswith(f"{event_type}_") for key in row)
            }
        if climate_zone:
            ranked = {
                run_id: row
                for run_id, row in ranked.items()
                if row.get("climate_zone") == climate_zone
            }

        results = sorted(
            ranked.values(),
            key=lambda row: (
                row.get(metric) is None,
                float(row.get(metric) or 0.0),
                str(row.get("model_name") or ""),
            ),
        )
        for rank, row in enumerate(results, start=1):
            row["rank"] = rank
        return {"results": results}

    return db_backed_response(query, {"results": []})


def create_submission(payload: SubmissionCreateRequest) -> SubmissionCreateResponse:
    db_iter, db = require_db()
    try:
        submission = db_models.Submission(
            model_id=payload.model_id,
            benchmark_task_id=payload.benchmark_task_id,
            name=payload.name or f"submission-model-{payload.model_id}",
            status="PENDING",
            submitted_by=payload.submitted_by,
        )
        db.add(submission)
        db.flush()

        stored_prediction_path = store_local_artifact(
            payload.prediction_csv_path,
            f"submissions/{submission.id}/prediction.csv",
        )
        submission.artifact_uri = stored_prediction_path
        db.add(
            db_models.Artifact(
                submission_id=submission.id,
                artifact_type="prediction_csv",
                uri=stored_prediction_path,
            )
        )

        evaluation_run = db_models.EvaluationRun(
            submission_id=submission.id,
            split_protocol_id=payload.split_protocol_id,
            status="PENDING",
        )
        db.add(evaluation_run)
        db.flush()

        db.commit()
        db.refresh(submission)
        db.refresh(evaluation_run)
        evaluate_submission_task.delay(evaluation_run.id)

        return SubmissionCreateResponse(
            submission_id=submission.id,
            evaluation_run_id=evaluation_run.id,
            status=evaluation_run.status,
            prediction_csv_path=submission.artifact_uri or payload.prediction_csv_path,
        )
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not create submission: {exc}",
        ) from exc
    except FileNotFoundError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not store submission artifact: {exc}",
        ) from exc
    finally:
        close_db_iter(db_iter)


def create_submission_from_upload(
    *,
    model_id: int,
    benchmark_task_id: int,
    split_protocol_id: int,
    prediction_csv: UploadFile,
    name: str | None = None,
    submitted_by: str | None = None,
) -> SubmissionCreateResponse:
    suffix = Path(prediction_csv.filename or "prediction.csv").suffix or ".csv"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(prediction_csv.file.read())
        tmp_path = tmp.name

    try:
        return create_submission(
            SubmissionCreateRequest(
                model_id=model_id,
                benchmark_task_id=benchmark_task_id,
                split_protocol_id=split_protocol_id,
                prediction_csv_path=tmp_path,
                name=name,
                submitted_by=submitted_by,
            )
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def get_submission(submission_id: int) -> SubmissionDetailResponse:
    db_iter, db = require_db()
    try:
        submission = db.get(db_models.Submission, submission_id)
        if submission is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found.")

        runs = db.scalars(
            select(db_models.EvaluationRun).where(db_models.EvaluationRun.submission_id == submission.id)
        ).all()
        return SubmissionDetailResponse(
            submission_id=submission.id,
            model_id=submission.model_id,
            benchmark_task_id=submission.benchmark_task_id,
            name=submission.name,
            status=submission.status,
            prediction_csv_path=submission.artifact_uri,
            evaluation_runs=[
                {
                    "evaluation_run_id": run.id,
                    "split_protocol_id": run.split_protocol_id,
                    "status": run.status,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                }
                for run in runs
            ],
        )
    finally:
        close_db_iter(db_iter)


def get_evaluation_run(run_id: int) -> EvaluationRunDetailResponse:
    db_iter, db = require_db()
    try:
        evaluation_run = db.get(db_models.EvaluationRun, run_id)
        if evaluation_run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found.")

        metrics = db.scalars(
            select(db_models.Metric).where(db_models.Metric.evaluation_run_id == evaluation_run.id)
        ).all()
        artifacts = db.scalars(
            select(db_models.Artifact).where(
                (db_models.Artifact.evaluation_run_id == evaluation_run.id)
                | (db_models.Artifact.submission_id == evaluation_run.submission_id)
            )
        ).all()
        return EvaluationRunDetailResponse(
            evaluation_run_id=evaluation_run.id,
            submission_id=evaluation_run.submission_id,
            split_protocol_id=evaluation_run.split_protocol_id,
            status=evaluation_run.status,
            error_message=evaluation_run.logs_uri,
            metrics=[
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                }
                for metric in metrics
            ],
            artifacts=[
                {
                    "artifact_id": artifact.id,
                    "artifact_type": artifact.artifact_type,
                    "uri": artifact.uri,
                }
                for artifact in artifacts
            ],
        )
    finally:
        close_db_iter(db_iter)


def get_evaluation_run_status(run_id: int) -> EvaluationRunStatusResponse:
    db_iter, db = require_db()
    try:
        evaluation_run = db.get(db_models.EvaluationRun, run_id)
        if evaluation_run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found.")

        progress_by_status = {
            "PENDING": 5,
            "RUNNING": 55,
            "COMPLETED": 100,
            "FAILED": 100,
        }
        return EvaluationRunStatusResponse(
            evaluation_run_id=evaluation_run.id,
            status=evaluation_run.status,
            progress_percent=progress_by_status.get(evaluation_run.status, 0),
            error_message=evaluation_run.logs_uri,
        )
    finally:
        close_db_iter(db_iter)


def get_artifact_download_path(artifact_id: int) -> Path:
    db_iter, db = require_db()
    try:
        artifact = db.get(db_models.Artifact, artifact_id)
        if artifact is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.")

        artifact_path = Path(artifact.uri)
        if not artifact_path.exists() or not artifact_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found.")

        return artifact_path
    finally:
        close_db_iter(db_iter)
