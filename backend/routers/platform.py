"""Database-backed platform API routes."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import FileResponse

from backend.schemas import (
    EvaluationRunDetailResponse,
    EvaluationRunStatusResponse,
    SubmissionCreateRequest,
    SubmissionCreateResponse,
    SubmissionDetailResponse,
)
from backend.services import platform_service

router = APIRouter(prefix="/api", tags=["platform"])


@router.get("/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models")
def api_models() -> list[dict[str, object]]:
    return platform_service.list_models()


@router.get("/datasets")
def api_datasets() -> list[dict[str, object]]:
    return platform_service.list_datasets()


@router.get("/leaderboard")
def api_leaderboard(
    metric: str = "rmse",
    split_protocol: str | None = None,
    event_type: str | None = None,
    climate_zone: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    return platform_service.get_leaderboard(
        metric=metric,
        split_protocol=split_protocol,
        event_type=event_type,
        climate_zone=climate_zone,
    )


@router.post("/submissions", response_model=SubmissionCreateResponse, status_code=status.HTTP_201_CREATED)
def api_create_submission(payload: SubmissionCreateRequest) -> SubmissionCreateResponse:
    return platform_service.create_submission(payload)


@router.post("/submissions/upload", response_model=SubmissionCreateResponse, status_code=status.HTTP_201_CREATED)
def api_upload_submission(
    model_id: int = Form(...),
    benchmark_task_id: int = Form(...),
    split_protocol_id: int = Form(...),
    prediction_csv: UploadFile = File(...),
    name: str | None = Form(None),
    submitted_by: str | None = Form(None),
) -> SubmissionCreateResponse:
    return platform_service.create_submission_from_upload(
        model_id=model_id,
        benchmark_task_id=benchmark_task_id,
        split_protocol_id=split_protocol_id,
        prediction_csv=prediction_csv,
        name=name,
        submitted_by=submitted_by,
    )


@router.get("/submissions/{submission_id}", response_model=SubmissionDetailResponse)
def api_get_submission(submission_id: int) -> SubmissionDetailResponse:
    return platform_service.get_submission(submission_id)


@router.get("/evaluation-runs/{run_id}", response_model=EvaluationRunDetailResponse)
def api_get_evaluation_run(run_id: int) -> EvaluationRunDetailResponse:
    return platform_service.get_evaluation_run(run_id)


@router.get("/evaluation-runs/{run_id}/status", response_model=EvaluationRunStatusResponse)
def api_get_evaluation_run_status(run_id: int) -> EvaluationRunStatusResponse:
    return platform_service.get_evaluation_run_status(run_id)


@router.get("/artifacts/{artifact_id}/download")
def api_download_artifact(artifact_id: int) -> FileResponse:
    artifact_path = platform_service.get_artifact_download_path(artifact_id)
    return FileResponse(
        path=artifact_path,
        filename=artifact_path.name,
        media_type="application/octet-stream",
    )
