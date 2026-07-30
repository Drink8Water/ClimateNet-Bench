#!/usr/bin/env python
"""Run isolated benchmark seeds sequentially and aggregate their results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from climatenet.benchmark.multiseed import summarize_multiseed_runs
from climatenet.training.benchmark_runner import run_benchmark
from climatenet.utils.config import load_yaml
from climatenet.utils.paths import resolve_project_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_multiseed_config(config: dict) -> dict:
    seeds = [int(seed) for seed in config.get("random_seeds", [])]
    models = [
        model if isinstance(model, str) else model.get("name", "")
        for model in config.get("models", [])
    ]
    splits = list(config.get("split_protocols", []))
    feature_sets = list(config.get("feature_sets", {}))
    task_count = len(seeds) * len(models) * len(splits) * len(feature_sets)
    errors: list[str] = []
    if seeds != [42, 123, 2026]:
        errors.append(f"Expected seeds [42, 123, 2026], got {seeds}")
    if models != ["linear_regression", "lightgbm"]:
        errors.append(f"Unexpected models: {models}")
    if set(feature_sets) != {"full"}:
        errors.append(f"Only full feature set is allowed, got {feature_sets}")
    if splits != [
        "random_split",
        "temporal_holdout",
        "spatial_block_holdout",
    ]:
        errors.append(f"Unexpected split protocols: {splits}")
    if task_count != int(config.get("expected_total_tasks", 18)):
        errors.append(f"Expected 18 tasks, resolved {task_count}")
    full = config.get("feature_sets", {}).get("full", {}).get("features", [])
    if "dryness_proxy_log1p" not in full or "dryness_proxy" in full:
        errors.append("full must use dryness_proxy_log1p, not dryness_proxy")
    features_path = Path(config.get("features_path", ""))
    if not features_path.is_file():
        errors.append(f"Missing physical feature CSV: {features_path}")
    input_data_path = config.get("input_data_path")
    if input_data_path and Path(input_data_path) != features_path:
        errors.append("input_data_path and features_path must identify the same file")
    if config.get("synthetic") is not False:
        errors.append("formal multi-seed benchmark must set synthetic: false")
    if "synthetic" in str(features_path).lower():
        errors.append("features_path appears to reference synthetic data")
    expected_sha256 = config.get("expected_features_sha256")
    actual_sha256 = None
    if expected_sha256 and features_path.is_file():
        actual_sha256 = _sha256_file(features_path)
        if actual_sha256 != expected_sha256:
            errors.append(
                "features SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    audit_path_value = config.get("preflight_audit_path")
    audit_status = None
    if audit_path_value:
        audit_path = Path(audit_path_value)
        if not audit_path.is_file():
            errors.append(f"Missing preflight audit: {audit_path}")
        else:
            audit_status = json.loads(audit_path.read_text(encoding="utf-8")).get(
                "status"
            )
            required_status = config.get("required_preflight_audit_status")
            if required_status and audit_status != required_status:
                errors.append(
                    "preflight audit status mismatch: "
                    f"expected {required_status!r}, got {audit_status!r}"
                )
    existing_seed_runs: dict[int, str] = {
        int(seed): path for seed, path in config.get("existing_seed_runs", {}).items()
    }
    for seed, run_path in existing_seed_runs.items():
        metadata_path = Path(run_path) / "run_metadata.json"
        if not metadata_path.is_file():
            errors.append(f"Missing existing seed metadata: {metadata_path}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "completed":
            errors.append(f"Existing seed run is not completed: {run_path}")
        if int(metadata.get("seed", -1)) != seed:
            errors.append(
                f"Existing seed run mismatch: config seed {seed}, "
                f"metadata seed {metadata.get('seed')}"
            )
        if expected_sha256:
            hashes = metadata.get("data_files", {})
            if expected_sha256 not in hashes.values():
                errors.append(
                    f"Existing seed run does not reference expected SHA256: {run_path}"
                )
    artifacts = config.get("artifacts", {})
    estimate = int(artifacts.get("estimated_aggregate_bytes_including_reused_seed", 0))
    maximum = int(artifacts.get("estimated_max_bytes", 15_000_000_000))
    if estimate > maximum:
        errors.append(f"Artifact estimate exceeds limit: {estimate} > {maximum}")
    return {
        "status": "ready" if not errors else "error",
        "seeds": seeds,
        "models": models,
        "splits": splits,
        "feature_sets": feature_sets,
        "task_count": task_count,
        "features_path": str(features_path.resolve()),
        "features_sha256": actual_sha256,
        "expected_features_sha256": expected_sha256,
        "preflight_audit_path": audit_path_value,
        "preflight_audit_status": audit_status,
        "existing_seed_runs": existing_seed_runs,
        "artifact_estimate_bytes": estimate,
        "artifact_limit_bytes": maximum,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/benchmark/era5_land_v1_multiseed.yaml",
    )
    parser.add_argument("--dry-check", action="store_true")
    args = parser.parse_args()
    config = load_yaml(resolve_project_path(args.config))
    check = validate_multiseed_config(config)
    print(json.dumps(check, indent=2, ensure_ascii=False))
    if check["errors"]:
        raise SystemExit(2)
    if args.dry_check:
        return

    run_dirs: list[Path] = []
    existing = {
        int(seed): Path(path)
        for seed, path in config.get("existing_seed_runs", {}).items()
    }
    for seed in check["seeds"]:
        if seed in existing:
            run_dirs.append(existing[seed])
            continue
        seed_config = dict(config)
        seed_config["random_seed"] = seed
        seed_config.pop("random_seeds", None)
        seed_config.pop("existing_seed_runs", None)
        seed_config["benchmark_name"] = (
            f"{config['benchmark_name']}-seed{seed}"
        )
        registry = run_benchmark(
            seed_config,
            output_root=config["output_dir"],
        )
        if registry.list_failed():
            print(f"Seed {seed} completed with failed tasks; preserving run.")
        run_dirs.append(registry.path.parent)
    summary = summarize_multiseed_runs(
        run_dirs,
        config["summary_dir"],
        expected_seeds=check["seeds"],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
