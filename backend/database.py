"""Database connection helpers for the read-only FastAPI backend.

When DATABASE_URL is not set the backend runs in file-based mode.
The RuntimeError has been replaced with a warning so that ``import
backend.database`` does not prevent the FastAPI application from starting.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Generator
from typing import Any

from dotenv import load_dotenv
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


class Base(DeclarativeBase):
    """Base class for backend SQLAlchemy models."""

    pass

_engine: Any = None
_SessionLocal: Any = None
_db_available: bool = False


def _init_db() -> None:
    """Lazily initialise the database engine when DATABASE_URL is set."""
    global _engine, _SessionLocal, _db_available

    if _engine is not None:
        return  # already initialised

    if not DATABASE_URL:
        logger.info(
            "DATABASE_URL is not set — running in file-based mode. "
            "Copy .env.example to .env and edit it to enable PostgreSQL."
        )
        return

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
        _db_available = True
        logger.info("Database engine initialised.")
    except Exception as exc:
        logger.warning("Could not initialise database: %s — falling back to file-based mode.", exc)


def is_db_available() -> bool:
    """Return True when PostgreSQL is configured and reachable."""
    _init_db()
    return _db_available


def get_engine() -> Any:
    """Return the configured SQLAlchemy engine, or None in mock mode."""
    _init_db()
    return _engine


def get_db() -> Generator[Any, None, None]:
    """Yield a database session for FastAPI dependencies.

    Raises RuntimeError if the database is not configured.
    """
    _init_db()
    if not _db_available or _SessionLocal is None:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and edit it."
        )
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
