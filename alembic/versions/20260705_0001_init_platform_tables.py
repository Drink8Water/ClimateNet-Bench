"""init platform tables

Revision ID: 20260705_0001
Revises:
Create Date: 2026-07-05 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260705_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("storage_uri", sa.String(length=512), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_datasets_id", "datasets", ["id"])
    op.create_index("ix_datasets_name", "datasets", ["name"])

    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("model_type", sa.String(length=80), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_models_id", "models", ["id"])
    op.create_index("ix_models_name", "models", ["name"])

    op.create_table(
        "benchmark_tasks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("target_variable", sa.String(length=120), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.UniqueConstraint("dataset_id", "name", name="uq_benchmark_task_dataset_name"),
    )
    op.create_index("ix_benchmark_tasks_id", "benchmark_tasks", ["id"])
    op.create_index("ix_benchmark_tasks_dataset_id", "benchmark_tasks", ["dataset_id"])
    op.create_index("ix_benchmark_tasks_name", "benchmark_tasks", ["name"])

    op.create_table(
        "split_protocols",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("benchmark_task_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("protocol_type", sa.String(length=64), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["benchmark_task_id"], ["benchmark_tasks.id"]),
        sa.UniqueConstraint("benchmark_task_id", "name", name="uq_split_protocol_task_name"),
    )
    op.create_index("ix_split_protocols_id", "split_protocols", ["id"])
    op.create_index("ix_split_protocols_dataset_id", "split_protocols", ["dataset_id"])
    op.create_index("ix_split_protocols_benchmark_task_id", "split_protocols", ["benchmark_task_id"])
    op.create_index("ix_split_protocols_name", "split_protocols", ["name"])

    op.create_table(
        "submissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("benchmark_task_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("submitted_by", sa.String(length=120), nullable=True),
        sa.Column("artifact_uri", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"]),
        sa.ForeignKeyConstraint(["benchmark_task_id"], ["benchmark_tasks.id"]),
    )
    op.create_index("ix_submissions_id", "submissions", ["id"])
    op.create_index("ix_submissions_model_id", "submissions", ["model_id"])
    op.create_index("ix_submissions_benchmark_task_id", "submissions", ["benchmark_task_id"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("submission_id", sa.Integer(), nullable=False),
        sa.Column("split_protocol_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("logs_uri", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["submission_id"], ["submissions.id"]),
        sa.ForeignKeyConstraint(["split_protocol_id"], ["split_protocols.id"]),
    )
    op.create_index("ix_evaluation_runs_id", "evaluation_runs", ["id"])
    op.create_index("ix_evaluation_runs_submission_id", "evaluation_runs", ["submission_id"])
    op.create_index("ix_evaluation_runs_split_protocol_id", "evaluation_runs", ["split_protocol_id"])

    op.create_table(
        "metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("evaluation_run_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=40), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["evaluation_run_id"], ["evaluation_runs.id"]),
        sa.UniqueConstraint("evaluation_run_id", "name", name="uq_metric_run_name"),
    )
    op.create_index("ix_metrics_id", "metrics", ["id"])
    op.create_index("ix_metrics_evaluation_run_id", "metrics", ["evaluation_run_id"])
    op.create_index("ix_metrics_name", "metrics", ["name"])

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("submission_id", sa.Integer(), nullable=True),
        sa.Column("evaluation_run_id", sa.Integer(), nullable=True),
        sa.Column("artifact_type", sa.String(length=80), nullable=False),
        sa.Column("uri", sa.String(length=512), nullable=False),
        sa.Column("checksum", sa.String(length=128), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["submission_id"], ["submissions.id"]),
        sa.ForeignKeyConstraint(["evaluation_run_id"], ["evaluation_runs.id"]),
    )
    op.create_index("ix_artifacts_id", "artifacts", ["id"])
    op.create_index("ix_artifacts_submission_id", "artifacts", ["submission_id"])
    op.create_index("ix_artifacts_evaluation_run_id", "artifacts", ["evaluation_run_id"])


def downgrade() -> None:
    op.drop_index("ix_artifacts_evaluation_run_id", table_name="artifacts")
    op.drop_index("ix_artifacts_submission_id", table_name="artifacts")
    op.drop_index("ix_artifacts_id", table_name="artifacts")
    op.drop_table("artifacts")

    op.drop_index("ix_metrics_name", table_name="metrics")
    op.drop_index("ix_metrics_evaluation_run_id", table_name="metrics")
    op.drop_index("ix_metrics_id", table_name="metrics")
    op.drop_table("metrics")

    op.drop_index("ix_evaluation_runs_split_protocol_id", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_submission_id", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_id", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")

    op.drop_index("ix_submissions_benchmark_task_id", table_name="submissions")
    op.drop_index("ix_submissions_model_id", table_name="submissions")
    op.drop_index("ix_submissions_id", table_name="submissions")
    op.drop_table("submissions")

    op.drop_index("ix_split_protocols_name", table_name="split_protocols")
    op.drop_index("ix_split_protocols_benchmark_task_id", table_name="split_protocols")
    op.drop_index("ix_split_protocols_dataset_id", table_name="split_protocols")
    op.drop_index("ix_split_protocols_id", table_name="split_protocols")
    op.drop_table("split_protocols")

    op.drop_index("ix_benchmark_tasks_name", table_name="benchmark_tasks")
    op.drop_index("ix_benchmark_tasks_dataset_id", table_name="benchmark_tasks")
    op.drop_index("ix_benchmark_tasks_id", table_name="benchmark_tasks")
    op.drop_table("benchmark_tasks")

    op.drop_index("ix_models_name", table_name="models")
    op.drop_index("ix_models_id", table_name="models")
    op.drop_table("models")

    op.drop_index("ix_datasets_name", table_name="datasets")
    op.drop_index("ix_datasets_id", table_name="datasets")
    op.drop_table("datasets")
