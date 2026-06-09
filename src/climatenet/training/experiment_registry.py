"""Experiment metadata registry for ClimateNet-Bench.

Tracks every benchmark experiment so results can be aggregated into
a leaderboard without scanning every file in every directory.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    """One row in the experiment registry."""

    experiment_id: str
    benchmark_name: str = ""
    model_name: str = ""
    split_protocol: str = ""
    feature_set: str = ""
    train_regions: list[str] = field(default_factory=list)
    test_regions: list[str] = field(default_factory=list)
    train_years: list[int] = field(default_factory=list)
    test_years: list[int] = field(default_factory=list)
    timestamp: str = ""
    seed: int = 42
    status: str = "pending"  # pending | running | completed | failed
    metrics_path: str = ""
    predictions_path: str = ""
    intervals_path: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "split_protocol": self.split_protocol,
            "feature_set": self.feature_set,
            "train_regions": self.train_regions,
            "test_regions": self.test_regions,
            "train_years": self.train_years,
            "test_years": self.test_years,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "status": self.status,
            "metrics_path": self.metrics_path,
            "predictions_path": self.predictions_path,
            "intervals_path": self.intervals_path,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentRecord:
        return cls(
            experiment_id=str(d.get("experiment_id", "")),
            benchmark_name=str(d.get("benchmark_name", "")),
            model_name=str(d.get("model_name", "")),
            split_protocol=str(d.get("split_protocol", "")),
            feature_set=str(d.get("feature_set", "")),
            train_regions=list(d.get("train_regions", [])),
            test_regions=list(d.get("test_regions", [])),
            train_years=list(d.get("train_years", [])),
            test_years=list(d.get("test_years", [])),
            timestamp=str(d.get("timestamp", "")),
            seed=int(d.get("seed", 42)),
            status=str(d.get("status", "pending")),
            metrics_path=str(d.get("metrics_path", "")),
            predictions_path=str(d.get("predictions_path", "")),
            intervals_path=str(d.get("intervals_path", "")),
            error_message=str(d.get("error_message", "")),
        )


class ExperimentRegistry:
    """Registry of all experiments for a benchmark run.

    Saved as ``experiment_registry.json`` in the benchmark output root.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self.path = Path(registry_path)
        self._experiments: dict[str, ExperimentRecord] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, record: ExperimentRecord) -> None:
        self._experiments[record.experiment_id] = record

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self._experiments.get(experiment_id)

    def mark_running(self, experiment_id: str) -> None:
        if experiment_id in self._experiments:
            self._experiments[experiment_id].status = "running"
            self._experiments[experiment_id].timestamp = time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )

    def mark_completed(self, experiment_id: str) -> None:
        if experiment_id in self._experiments:
            self._experiments[experiment_id].status = "completed"

    def mark_failed(self, experiment_id: str, error: str) -> None:
        if experiment_id in self._experiments:
            self._experiments[experiment_id].status = "failed"
            self._experiments[experiment_id].error_message = error

    def list_all(self) -> list[ExperimentRecord]:
        return list(self._experiments.values())

    def list_completed(self) -> list[ExperimentRecord]:
        return [r for r in self._experiments.values() if r.status == "completed"]

    def list_failed(self) -> list[ExperimentRecord]:
        return [r for r in self._experiments.values() if r.status == "failed"]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "experiments": [r.to_dict() for r in self._experiments.values()]
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentRegistry:
        p = Path(path)
        registry = cls(p)
        if not p.exists():
            return registry
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("experiments", []):
            record = ExperimentRecord.from_dict(entry)
            registry.add(record)
        return registry
