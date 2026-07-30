#!/usr/bin/env python
"""Run the ClimateNet-Bench benchmark pipeline.

Usage
-----
.. code-block:: bash

    # Smoke test (fast)
    python scripts/run_benchmark.py --config configs/benchmark/smoke_test.yaml

    # Full benchmark
    python scripts/run_benchmark.py --config configs/benchmark/evap_anomaly_v1.yaml

    # Custom output directory
    python scripts/run_benchmark.py --config configs/benchmark/smoke_test.yaml \\
        --output-dir outputs/benchmark_runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.training.benchmark_runner import run_benchmark
from climatenet.utils.config import load_yaml
from climatenet.utils.paths import resolve_project_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_benchmark")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClimateNet-Bench benchmark")
    parser.add_argument(
        "--config",
        default="configs/benchmark/smoke_test.yaml",
        help="Path to benchmark YAML config",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root directory for benchmark outputs; overrides config output_dir",
    )
    parser.add_argument(
        "--dry-check",
        action="store_true",
        help="Validate the configured task matrix and artifact budget without running",
    )
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    logger.info("Loading config from %s", config_path)
    config = load_yaml(config_path)
    output_dir = args.output_dir or config.get(
        "output_dir", "outputs/benchmark_runs"
    )

    logger.info("Benchmark: %s", config.get("benchmark_name", "unnamed"))
    logger.info("Models: %d", len(config.get("models", [])))
    logger.info("Output: %s", output_dir)

    if args.dry_check:
        models = [
            model if isinstance(model, str) else model.get("name", "")
            for model in config.get("models", [])
        ]
        splits = list(config.get("split_protocols", []))
        feature_sets = list(config.get("feature_sets", {}))
        repeated = config.get("repeated_spatial", {})
        repeated_folds = (
            list(repeated.get("folds", []))
            if "repeated_region_stratified_spatial" in splits
            else []
        )
        resolved_split_count = (
            len(repeated_folds) if repeated_folds else len(splits)
        )
        task_count = len(models) * resolved_split_count * len(feature_sets)
        features_path = Path(config.get("features_path", ""))
        forbidden = {
            "tcn",
            "xgboost",
            "random_forest",
        } & set(models)
        forbidden_splits = {
            "region_transfer",
            "climate_zone_transfer",
            "spatiotemporal_holdout",
            "spatial_temporal_holdout",
        } & set(splits)
        errors: list[str] = []
        if config.get("synthetic") is not False:
            errors.append("formal v1 must set synthetic: false")
        if "synthetic" in str(features_path).lower():
            errors.append("features_path appears to reference synthetic data")
        if not features_path.is_file():
            errors.append(f"features_path does not exist: {features_path}")
        input_data_path = config.get("input_data_path")
        if input_data_path and Path(input_data_path) != features_path:
            errors.append(
                "input_data_path and features_path must identify the same file"
            )
        expected_sha256 = config.get("expected_features_sha256")
        actual_sha256 = (
            _sha256_file(features_path)
            if expected_sha256 and features_path.is_file()
            else None
        )
        if expected_sha256 and actual_sha256 != expected_sha256:
            errors.append(
                "features SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        audit_path_value = config.get("preflight_audit_path")
        audit_status = None
        if audit_path_value:
            audit_path = Path(audit_path_value)
            if not audit_path.is_file():
                errors.append(f"preflight audit does not exist: {audit_path}")
            else:
                try:
                    audit_status = json.loads(
                        audit_path.read_text(encoding="utf-8")
                    ).get("status")
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"cannot read preflight audit: {exc}")
                required_status = config.get(
                    "required_preflight_audit_status"
                )
                if required_status and audit_status != required_status:
                    errors.append(
                        "preflight audit status mismatch: "
                        f"expected {required_status!r}, got {audit_status!r}"
                    )
        if forbidden:
            errors.append(f"forbidden models configured: {sorted(forbidden)}")
        if forbidden_splits:
            errors.append(
                f"forbidden split protocols configured: {sorted(forbidden_splits)}"
            )
        expected_task_count = int(config.get("expected_task_count", 12))
        if task_count != expected_task_count:
            errors.append(
                f"expected exactly {expected_task_count} tasks, "
                f"resolved {task_count}"
            )
        fold_audit_path = repeated.get("audit_path")
        fold_audit_status = None
        fold_audit_sha256 = None
        if repeated_folds:
            if len(repeated_folds) != len(set(repeated_folds)):
                errors.append("repeated spatial folds contain duplicates")
            if int(config.get("fold_count", -1)) != len(repeated_folds):
                errors.append("fold_count does not match repeated_spatial.folds")
            if not fold_audit_path:
                errors.append("repeated spatial audit_path is required")
            else:
                fold_audit = Path(fold_audit_path)
                if not fold_audit.is_file():
                    errors.append(
                        f"fold audit does not exist: {fold_audit}"
                    )
                else:
                    try:
                        fold_payload = json.loads(
                            fold_audit.read_text(encoding="utf-8")
                        )
                        fold_audit_status = fold_payload.get("status")
                        fold_audit_sha256 = _sha256_file(fold_audit)
                        required = repeated.get(
                            "required_audit_status", "ready"
                        )
                        if (
                            fold_audit_status != required
                            or not fold_payload.get("audit_passed", False)
                        ):
                            errors.append(
                                "fold audit is not accepted: "
                                f"status={fold_audit_status!r}"
                            )
                    except (OSError, json.JSONDecodeError) as exc:
                        errors.append(f"cannot read fold audit: {exc}")
            manifest_dir = Path(repeated.get("manifest_dir", ""))
            for fold in repeated_folds:
                manifest = (
                    manifest_dir
                    / f"fold_{int(fold)}_block_assignments.csv"
                )
                if not manifest.is_file():
                    errors.append(f"fold manifest does not exist: {manifest}")
            if config.get("artifacts", {}).get(
                "compatibility_predictions", True
            ):
                errors.append(
                    "repeated spatial run must disable compatibility predictions"
                )
        artifact_config = config.get("artifacts", {})
        estimated_bytes = int(
            artifact_config.get(
                "estimated_total_run_bytes",
                artifact_config.get(
                    "estimated_total_bytes",
                    artifact_config.get("estimated_prediction_bytes", 0),
                ),
            )
        )
        maximum_bytes = int(
            artifact_config.get("estimated_max_bytes", 10_000_000_000)
        )
        if estimated_bytes > maximum_bytes:
            errors.append(
                "estimated prediction artifacts exceed configured maximum: "
                f"{estimated_bytes} > {maximum_bytes}"
            )
        report = {
            "status": "ready" if not errors else "error",
            "task_count": task_count,
            "expected_task_count": expected_task_count,
            "models": models,
            "split_protocols": splits,
            "feature_sets": feature_sets,
            "features_path": str(features_path.resolve()),
            "features_size_bytes": (
                features_path.stat().st_size if features_path.is_file() else None
            ),
            "features_sha256": actual_sha256,
            "expected_features_sha256": expected_sha256,
            "preflight_audit_path": audit_path_value,
            "preflight_audit_status": audit_status,
            "repeated_spatial_folds": repeated_folds,
            "fold_audit_path": fold_audit_path,
            "fold_audit_status": fold_audit_status,
            "fold_audit_sha256": fold_audit_sha256,
            "output_root": str(Path(output_dir).resolve()),
            "git_state_will_be_recorded_in_run_metadata": True,
            "artifacts": artifact_config,
            "errors": errors,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if errors:
            raise SystemExit(2)
        return

    registry = run_benchmark(config=config, output_root=output_dir)

    completed = registry.list_completed()
    failed = registry.list_failed()
    logger.info("=== Benchmark Complete ===")
    logger.info("  Completed: %d", len(completed))
    logger.info("  Failed:    %d", len(failed))
    if failed:
        for r in failed:
            logger.info("    ❌ %s: %s", r.experiment_id, r.error_message)


if __name__ == "__main__":
    main()
