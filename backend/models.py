"""SQLAlchemy schema for the ClimateNet-Bench evaluation platform."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class Dataset(Base):
    """Benchmark dataset metadata."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    description: Mapped[str | None] = mapped_column(Text)
    storage_uri: Mapped[str | None] = mapped_column(String(512))
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    benchmark_tasks: Mapped[list["BenchmarkTask"]] = relationship(back_populates="dataset")
    split_protocols: Mapped[list["SplitProtocol"]] = relationship(back_populates="dataset")


class BenchmarkTask(Base):
    """A target prediction/evaluation task defined over a dataset."""

    __tablename__ = "benchmark_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    target_variable: Mapped[str] = mapped_column(String(120), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False, default="regression")
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    dataset: Mapped["Dataset"] = relationship(back_populates="benchmark_tasks")
    split_protocols: Mapped[list["SplitProtocol"]] = relationship(back_populates="benchmark_task")
    submissions: Mapped[list["Submission"]] = relationship(back_populates="benchmark_task")

    __table_args__ = (UniqueConstraint("dataset_id", "name", name="uq_benchmark_task_dataset_name"),)


class SplitProtocol(Base):
    """Train/test split protocol attached to a benchmark task."""

    __tablename__ = "split_protocols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    benchmark_task_id: Mapped[int] = mapped_column(ForeignKey("benchmark_tasks.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    protocol_type: Mapped[str] = mapped_column(String(64), nullable=False)
    config: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    dataset: Mapped["Dataset"] = relationship(back_populates="split_protocols")
    benchmark_task: Mapped["BenchmarkTask"] = relationship(back_populates="split_protocols")
    evaluation_runs: Mapped[list["EvaluationRun"]] = relationship(back_populates="split_protocol")

    __table_args__ = (UniqueConstraint("benchmark_task_id", "name", name="uq_split_protocol_task_name"),)


class BenchmarkModel(Base):
    """Registered model metadata for submissions."""

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    model_type: Mapped[str | None] = mapped_column(String(80))
    description: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    submissions: Mapped[list["Submission"]] = relationship(back_populates="model")


class Submission(Base):
    """A model submission for a benchmark task."""

    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False, index=True)
    benchmark_task_id: Mapped[int] = mapped_column(ForeignKey("benchmark_tasks.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="pending")
    submitted_by: Mapped[str | None] = mapped_column(String(120))
    artifact_uri: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    model: Mapped["BenchmarkModel"] = relationship(back_populates="submissions")
    benchmark_task: Mapped["BenchmarkTask"] = relationship(back_populates="submissions")
    evaluation_runs: Mapped[list["EvaluationRun"]] = relationship(back_populates="submission")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="submission")


class EvaluationRun(Base):
    """Execution record for evaluating a submission on a split protocol."""

    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"), nullable=False, index=True)
    split_protocol_id: Mapped[int] = mapped_column(ForeignKey("split_protocols.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="pending")
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    logs_uri: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    submission: Mapped["Submission"] = relationship(back_populates="evaluation_runs")
    split_protocol: Mapped["SplitProtocol"] = relationship(back_populates="evaluation_runs")
    metrics: Mapped[list["Metric"]] = relationship(back_populates="evaluation_run")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="evaluation_run")


class Metric(Base):
    """Scalar metric produced by an evaluation run."""

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    evaluation_run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    evaluation_run: Mapped["EvaluationRun"] = relationship(back_populates="metrics")

    __table_args__ = (UniqueConstraint("evaluation_run_id", "name", name="uq_metric_run_name"),)


class Artifact(Base):
    """Stored artifact linked to a submission or evaluation run."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    submission_id: Mapped[int | None] = mapped_column(ForeignKey("submissions.id"), index=True)
    evaluation_run_id: Mapped[int | None] = mapped_column(ForeignKey("evaluation_runs.id"), index=True)
    artifact_type: Mapped[str] = mapped_column(String(80), nullable=False)
    uri: Mapped[str] = mapped_column(String(512), nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(128))
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    submission: Mapped["Submission | None"] = relationship(back_populates="artifacts")
    evaluation_run: Mapped["EvaluationRun | None"] = relationship(back_populates="artifacts")
