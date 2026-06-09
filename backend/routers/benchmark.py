"""Benchmark overview endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.services.benchmark_service import (
    get_benchmark_regions,
    get_benchmark_splits,
    get_benchmark_summary,
    get_benchmark_task,
)

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.get("/summary")
def benchmark_summary():
    """Return benchmark metadata: experiment counts, best model, split difficulty."""
    return get_benchmark_summary()


@router.get("/task")
def benchmark_task():
    """Return the benchmark task definition."""
    return get_benchmark_task()


@router.get("/regions")
def benchmark_regions():
    """Return the five benchmark region definitions."""
    return get_benchmark_regions()


@router.get("/splits")
def benchmark_splits():
    """Return available split protocol metadata."""
    return get_benchmark_splits()
