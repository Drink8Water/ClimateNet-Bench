"""Path helpers for ClimateNet."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the project root from inside the src/climatenet package."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
