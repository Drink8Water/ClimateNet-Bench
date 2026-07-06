"""Local filesystem artifact storage."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "data/artifacts"))


def store_local_artifact(source_path: str, relative_path: str) -> str:
    """Copy an artifact into local storage and return its stored path."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Artifact source not found: {source_path}")

    destination = ARTIFACT_ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return str(destination)
