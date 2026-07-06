"""Background tasks for submission evaluation."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError

from backend import models as db_models
from backend.celery_app import task
from backend.database import get_db
from backend.evaluation import evaluate_prediction_csv


@task
def evaluate_submission_task(evaluation_run_id: int) -> None:
    """Evaluate a submitted prediction CSV and persist metrics."""
    db_iter = get_db()
    db = next(db_iter)
    try:
        evaluation_run = db.get(db_models.EvaluationRun, evaluation_run_id)
        if evaluation_run is None:
            return

        evaluation_run.status = "RUNNING"
        evaluation_run.started_at = datetime.utcnow()
        db.commit()
        db.refresh(evaluation_run)

        submission = db.get(db_models.Submission, evaluation_run.submission_id)
        if submission is None or not submission.artifact_uri:
            raise ValueError("Evaluation run has no submission prediction CSV.")

        metric_values = evaluate_prediction_csv(submission.artifact_uri)
        db.execute(delete(db_models.Metric).where(db_models.Metric.evaluation_run_id == evaluation_run.id))
        for metric_name, metric_value in metric_values.items():
            db.add(
                db_models.Metric(
                    evaluation_run_id=evaluation_run.id,
                    name=metric_name,
                    value=metric_value,
                )
            )

        evaluation_run.status = "COMPLETED"
        evaluation_run.finished_at = datetime.utcnow()
        db.commit()
    except (FileNotFoundError, ValueError, SQLAlchemyError) as exc:
        db.rollback()
        evaluation_run = db.get(db_models.EvaluationRun, evaluation_run_id)
        if evaluation_run is not None:
            evaluation_run.status = "FAILED"
            evaluation_run.finished_at = datetime.utcnow()
            evaluation_run.logs_uri = str(exc)
            db.commit()
    finally:
        try:
            next(db_iter)
        except StopIteration:
            pass
