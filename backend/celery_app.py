"""Celery application configuration for backend evaluation tasks."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


_executor = ThreadPoolExecutor(max_workers=2)


class LocalAsyncTask:
    """Small Celery-compatible fallback used when Celery is unavailable."""

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.__name__ = func.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def delay(self, *args: Any, **kwargs: Any):
        return _executor.submit(self.func, *args, **kwargs)


try:
    from celery import Celery

    celery_app = Celery(
        "climatenet_backend",
        broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
        include=["backend.tasks"],
    )
    celery_app.conf.update(task_track_started=True)
except ModuleNotFoundError:
    celery_app = None


def task(func: Callable[..., Any]):
    """Register a Celery task, or a local async fallback in test/dev mode."""
    if celery_app is None:
        return LocalAsyncTask(func)
    return celery_app.task(name=f"backend.tasks.{func.__name__}")(func)
