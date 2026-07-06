from __future__ import annotations

from io import BytesIO
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import UploadFile
from sqlalchemy import inspect
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.routers import platform
from backend.main import (
    api_create_submission,
    api_datasets,
    api_download_artifact,
    api_get_evaluation_run,
    api_get_evaluation_run_status,
    api_get_submission,
    api_health,
    api_leaderboard,
    api_models,
    api_upload_submission,
    app,
)
from backend import models as db_models
from backend.schemas import SubmissionCreateRequest
from backend.seed import main as seed_main
from backend.seed import seed_platform_defaults
from backend.tasks import evaluate_submission_task


def _route_paths() -> set[str]:
    app_paths = {route.path for route in app.routes if hasattr(route, "path")}
    platform_paths = {route.path for route in platform.router.routes if hasattr(route, "path")}
    return app_paths | platform_paths


def test_api_health_without_database_url():
    route_paths = _route_paths()

    assert "/api/health" in route_paths
    assert api_health() == {"status": "ok"}


def test_mock_api_endpoints_without_database_url():
    route_paths = _route_paths()

    assert {"/api/models", "/api/datasets", "/api/leaderboard"}.issubset(route_paths)
    assert api_models() == []
    assert api_datasets() == []
    assert api_leaderboard() == {"results": []}


def test_backend_models_create_in_sqlite_memory_db():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(bind=engine)

    table_names = set(Base.metadata.tables)
    assert {
        "datasets",
        "benchmark_tasks",
        "split_protocols",
        "models",
        "submissions",
        "evaluation_runs",
        "metrics",
        "artifacts",
    }.issubset(table_names)

    assert db_models.Dataset.__tablename__ == "datasets"


def test_alembic_upgrade_creates_platform_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "alembic_platform.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    config = Config("alembic.ini")
    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    table_names = set(inspect(engine).get_table_names())
    assert {
        "datasets",
        "benchmark_tasks",
        "split_protocols",
        "models",
        "submissions",
        "evaluation_runs",
        "metrics",
        "artifacts",
    }.issubset(table_names)
    assert "alembic_version" in table_names


def test_seed_platform_defaults_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with SessionLocal() as db:
        first = seed_platform_defaults(db)
        second = seed_platform_defaults(db)

        assert first == second
        assert first == {
            "dataset_id": 1,
            "benchmark_task_id": 1,
            "split_protocol_id": 1,
            "model_id": 1,
        }

        assert len(db.scalars(select(db_models.Dataset)).all()) == 1
        assert len(db.scalars(select(db_models.BenchmarkTask)).all()) == 1
        assert len(db.scalars(select(db_models.SplitProtocol)).all()) == 1
        assert len(db.scalars(select(db_models.BenchmarkModel)).all()) == 1


def test_seed_cli_reports_missing_database(monkeypatch, capsys):
    monkeypatch.setattr("backend.seed.get_engine", lambda: None)

    exit_code = seed_main()

    assert exit_code == 1
    assert "DATABASE_URL is required" in capsys.readouterr().out


def test_seed_cli_seeds_configured_database(monkeypatch, capsys):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def get_test_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr("backend.seed.get_engine", lambda: engine)
    monkeypatch.setattr("backend.seed.get_db", get_test_db)

    exit_code = seed_main()

    assert exit_code == 0
    assert "dataset_id=1" in capsys.readouterr().out

    with SessionLocal() as db:
        assert db.scalar(select(db_models.Dataset).where(db_models.Dataset.name == "climatenet-demo")) is not None


def test_submission_creation_generates_metrics(monkeypatch, tmp_path):
    assert "/api/submissions" in _route_paths()

    prediction_csv = tmp_path / "prediction.csv"
    prediction_csv.write_text(
        "actual,prediction\n"
        "1.0,1.0\n"
        "2.0,2.5\n"
        "3.0,2.0\n",
        encoding="utf-8",
    )
    better_prediction_csv = tmp_path / "better_prediction.csv"
    better_prediction_csv.write_text(
        "actual,prediction\n"
        "1.0,1.0\n"
        "2.0,2.0\n"
        "3.0,3.0\n",
        encoding="utf-8",
    )

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with SessionLocal() as db:
        dataset = db_models.Dataset(name="smoke-dataset", version="v1")
        db.add(dataset)
        db.flush()
        task = db_models.BenchmarkTask(
            dataset_id=dataset.id,
            name="evap-anomaly",
            target_variable="evaporation_anomaly",
        )
        model = db_models.BenchmarkModel(name="baseline", version="v1")
        better_model = db_models.BenchmarkModel(name="perfect", version="v1")
        db.add_all([task, model, better_model])
        db.flush()
        split = db_models.SplitProtocol(
            dataset_id=dataset.id,
            benchmark_task_id=task.id,
            name="random",
            protocol_type="random",
        )
        db.add(split)
        db.commit()
        model_id = model.id
        better_model_id = better_model.id
        task_id = task.id
        split_id = split.id

    monkeypatch.setattr("backend.artifacts.ARTIFACT_ROOT", tmp_path / "artifacts")
    def get_test_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr("backend.services.platform_service.is_db_available", lambda: True)
    monkeypatch.setattr("backend.services.platform_service.get_db", get_test_db)
    monkeypatch.setattr("backend.tasks.get_db", get_test_db)

    class ImmediateEvaluationTask:
        called_with: list[int] = []

        def delay(self, evaluation_run_id: int):
            self.called_with.append(evaluation_run_id)
            return evaluate_submission_task(evaluation_run_id)

    immediate_task = ImmediateEvaluationTask()
    monkeypatch.setattr("backend.services.platform_service.evaluate_submission_task", immediate_task)

    response = api_create_submission(
        SubmissionCreateRequest(
            model_id=model_id,
            benchmark_task_id=task_id,
            split_protocol_id=split_id,
            prediction_csv_path=str(prediction_csv),
            name="baseline-submit",
        )
    )

    assert response.status == "PENDING"
    assert response.prediction_csv_path.endswith("submissions/1/prediction.csv")
    assert immediate_task.called_with == [response.evaluation_run_id]

    better_response = api_create_submission(
        SubmissionCreateRequest(
            model_id=better_model_id,
            benchmark_task_id=task_id,
            split_protocol_id=split_id,
            prediction_csv_path=str(better_prediction_csv),
            name="perfect-submit",
        )
    )
    assert better_response.status == "PENDING"
    assert immediate_task.called_with == [response.evaluation_run_id, better_response.evaluation_run_id]

    with SessionLocal() as db:
        submission = db.get(db_models.Submission, response.submission_id)
        evaluation_run = db.get(db_models.EvaluationRun, response.evaluation_run_id)
        better_run = db.get(db_models.EvaluationRun, better_response.evaluation_run_id)
        metrics = db.scalars(
            select(db_models.Metric).where(
                db_models.Metric.evaluation_run_id == response.evaluation_run_id
            )
        ).all()
        artifacts = db.scalars(
            select(db_models.Artifact).where(
                db_models.Artifact.submission_id == response.submission_id
            )
        ).all()

    assert submission is not None
    assert submission.artifact_uri is not None
    assert submission.artifact_uri.endswith("submissions/1/prediction.csv")
    assert submission.status == "PENDING"
    assert len(artifacts) == 1
    assert artifacts[0].artifact_type == "prediction_csv"
    assert evaluation_run is not None
    assert evaluation_run.submission_id == submission.id
    assert evaluation_run.split_protocol_id == split_id
    assert evaluation_run.status == "COMPLETED"
    assert {metric.name for metric in metrics} == {"mae", "rmse", "r2"}
    assert all(metric.value >= 0 for metric in metrics if metric.name != "r2")

    submission_detail = api_get_submission(response.submission_id)
    assert submission_detail.submission_id == response.submission_id
    assert submission_detail.evaluation_runs[0]["status"] == "COMPLETED"

    run_status = api_get_evaluation_run_status(response.evaluation_run_id)
    assert run_status.status == "COMPLETED"
    assert run_status.progress_percent == 100

    run_detail = api_get_evaluation_run(response.evaluation_run_id)
    assert run_detail.status == "COMPLETED"
    assert {metric["name"] for metric in run_detail.metrics} == {"mae", "rmse", "r2"}
    assert [artifact["artifact_type"] for artifact in run_detail.artifacts] == ["prediction_csv"]

    leaderboard = api_leaderboard()
    assert [row["model_name"] for row in leaderboard["results"]] == ["perfect", "baseline"]
    assert [row["rank"] for row in leaderboard["results"]] == [1, 2]
    assert leaderboard["results"][0]["rmse"] == 0.0


def test_submission_upload_creates_db_records_and_metrics(monkeypatch, tmp_path):
    assert "/api/submissions/upload" in _route_paths()

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with SessionLocal() as db:
        dataset = db_models.Dataset(name="upload-dataset", version="v1")
        db.add(dataset)
        db.flush()
        task = db_models.BenchmarkTask(
            dataset_id=dataset.id,
            name="upload-task",
            target_variable="evaporation_anomaly",
        )
        model = db_models.BenchmarkModel(name="upload-model", version="v1")
        db.add_all([task, model])
        db.flush()
        split = db_models.SplitProtocol(
            dataset_id=dataset.id,
            benchmark_task_id=task.id,
            name="upload-split",
            protocol_type="random",
        )
        db.add(split)
        db.commit()
        model_id = model.id
        task_id = task.id
        split_id = split.id

    monkeypatch.setattr("backend.artifacts.ARTIFACT_ROOT", tmp_path / "artifacts")

    def get_test_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr("backend.services.platform_service.is_db_available", lambda: True)
    monkeypatch.setattr("backend.services.platform_service.get_db", get_test_db)
    monkeypatch.setattr("backend.tasks.get_db", get_test_db)

    class ImmediateEvaluationTask:
        def delay(self, evaluation_run_id: int):
            return evaluate_submission_task(evaluation_run_id)

    monkeypatch.setattr("backend.services.platform_service.evaluate_submission_task", ImmediateEvaluationTask())

    response = api_upload_submission(
        model_id=model_id,
        benchmark_task_id=task_id,
        split_protocol_id=split_id,
        prediction_csv=UploadFile(
            file=BytesIO(b"actual,prediction\n1.0,1.0\n2.0,2.2\n3.0,2.8\n"),
            filename="prediction.csv",
        ),
        name="upload-submit",
        submitted_by=None,
    )

    assert response.status == "PENDING"
    assert response.prediction_csv_path.endswith("submissions/1/prediction.csv")

    with SessionLocal() as db:
        submission = db.get(db_models.Submission, response.submission_id)
        evaluation_run = db.get(db_models.EvaluationRun, response.evaluation_run_id)
        metrics = db.scalars(
            select(db_models.Metric).where(
                db_models.Metric.evaluation_run_id == response.evaluation_run_id
            )
        ).all()
        artifacts = db.scalars(
            select(db_models.Artifact).where(
                db_models.Artifact.submission_id == response.submission_id
            )
        ).all()

    assert submission is not None
    assert submission.name == "upload-submit"
    assert submission.artifact_uri is not None
    assert evaluation_run is not None
    assert evaluation_run.status == "COMPLETED"
    assert {metric.name for metric in metrics} == {"mae", "rmse", "r2"}
    assert len(artifacts) == 1
    assert artifacts[0].uri == submission.artifact_uri


def test_submission_with_event_columns_generates_detection_metrics(monkeypatch, tmp_path):
    prediction_csv = tmp_path / "event_prediction.csv"
    prediction_csv.write_text(
        "actual,prediction,soil_moisture_drought,soil_moisture_drought_pred\n"
        "1.0,1.0,1,1\n"
        "2.0,2.4,1,0\n"
        "3.0,2.8,0,1\n"
        "4.0,4.1,0,0\n",
        encoding="utf-8",
    )

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with SessionLocal() as db:
        dataset = db_models.Dataset(name="event-dataset", version="v1")
        db.add(dataset)
        db.flush()
        task = db_models.BenchmarkTask(
            dataset_id=dataset.id,
            name="event-task",
            target_variable="evaporation_anomaly",
        )
        model = db_models.BenchmarkModel(name="event-model", version="v1")
        db.add_all([task, model])
        db.flush()
        split = db_models.SplitProtocol(
            dataset_id=dataset.id,
            benchmark_task_id=task.id,
            name="event-split",
            protocol_type="random",
        )
        db.add(split)
        db.commit()
        model_id = model.id
        task_id = task.id
        split_id = split.id

    monkeypatch.setattr("backend.artifacts.ARTIFACT_ROOT", tmp_path / "artifacts")

    def get_test_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr("backend.services.platform_service.is_db_available", lambda: True)
    monkeypatch.setattr("backend.services.platform_service.get_db", get_test_db)
    monkeypatch.setattr("backend.tasks.get_db", get_test_db)

    class ImmediateEvaluationTask:
        def delay(self, evaluation_run_id: int):
            return evaluate_submission_task(evaluation_run_id)

    monkeypatch.setattr("backend.services.platform_service.evaluate_submission_task", ImmediateEvaluationTask())

    response = api_create_submission(
        SubmissionCreateRequest(
            model_id=model_id,
            benchmark_task_id=task_id,
            split_protocol_id=split_id,
            prediction_csv_path=str(prediction_csv),
            name="event-submit",
        )
    )

    with SessionLocal() as db:
        metrics = db.scalars(
            select(db_models.Metric).where(
                db_models.Metric.evaluation_run_id == response.evaluation_run_id
            )
        ).all()

    metric_values = {metric.name: metric.value for metric in metrics}
    assert {"mae", "rmse", "r2"}.issubset(metric_values)
    assert metric_values["soil_moisture_drought_pod"] == 0.5
    assert metric_values["soil_moisture_drought_far"] == 0.5
    assert metric_values["soil_moisture_drought_csi"] == 1 / 3
    assert abs(metric_values["soil_moisture_drought_intensity_bias"] - (1.7 / 1.5)) < 1e-12

    leaderboard = api_leaderboard(
        metric="soil_moisture_drought_csi",
        split_protocol="event-split",
        event_type="soil_moisture_drought",
    )
    assert len(leaderboard["results"]) == 1
    assert leaderboard["results"][0]["model_name"] == "event-model"
    assert leaderboard["results"][0]["soil_moisture_drought_csi"] == 1 / 3


def test_artifact_download_returns_local_file_response(monkeypatch, tmp_path):
    assert "/api/artifacts/{artifact_id}/download" in _route_paths()

    artifact_file = tmp_path / "prediction.csv"
    artifact_file.write_text("actual,prediction\n1,1\n", encoding="utf-8")

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with SessionLocal() as db:
        artifact = db_models.Artifact(
            artifact_type="prediction_csv",
            uri=str(artifact_file),
        )
        db.add(artifact)
        db.commit()
        artifact_id = artifact.id

    def get_test_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr("backend.services.platform_service.is_db_available", lambda: True)
    monkeypatch.setattr("backend.services.platform_service.get_db", get_test_db)

    response = api_download_artifact(artifact_id)

    assert Path(response.path) == artifact_file
    assert response.filename == "prediction.csv"
