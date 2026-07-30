"""Combined audit for corrected raw, processed, and physical ERA5 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climatenet.data.radiation_consistency import sha256_file

VARIABLE_MAP = {
    "radiation": "ssrd",
    "precipitation": "tp",
    "evaporation": "e",
}


def yearly_accumulated_means(
    path: str | Path, *, chunksize: int = 250_000
) -> tuple[pd.DataFrame, int, int]:
    source = Path(path)
    sums: dict[tuple[str, int, str], float] = {}
    counts: dict[tuple[str, int, str], int] = {}
    rows = 0
    negative_evaporation = 0
    columns = [
        "region",
        "year",
        "radiation",
        "precipitation",
        "evaporation",
    ]
    for chunk in pd.read_csv(source, usecols=columns, chunksize=chunksize):
        rows += len(chunk)
        negative_evaporation += int((chunk["evaporation"] < 0).sum())
        for keys, group in chunk.groupby(["region", "year"], observed=True):
            region, year = str(keys[0]), int(keys[1])
            for variable in VARIABLE_MAP:
                values = group[variable].to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                key = (region, year, variable)
                sums[key] = sums.get(key, 0.0) + float(finite.sum())
                counts[key] = counts.get(key, 0) + int(len(finite))
    frame = pd.DataFrame(
        [
            {
                "region": region,
                "year": year,
                "variable": variable,
                "mean": sums[key] / counts[key],
            }
            for key in sorted(sums)
            for region, year, variable in [key]
        ]
    )
    return frame, rows, negative_evaporation


def build_corrected_full_audit(config: dict[str, Any]) -> dict[str, Any]:
    era5 = config["era5"]
    paths = {
        "raw": Path(era5["raw_audit_path"]),
        "processed": Path(era5["processed_audit_path"]),
        "physical": Path(config["physical_features_audit_path"]),
        "patch": Path(config["provenance"]["patch_audit"]),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Corrected audit inputs missing: {missing}")
    reports = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in paths.items()
    }
    corrected, corrected_rows, corrected_negative = yearly_accumulated_means(
        era5["processed_path"]
    )
    old, old_rows, old_negative = yearly_accumulated_means(
        config["provenance"]["old_processed_csv"]
    )
    comparison = old.merge(
        corrected,
        on=["region", "year", "variable"],
        suffixes=("_old_bad", "_corrected"),
        validate="one_to_one",
    )
    comparison["absolute_change"] = (
        comparison["mean_corrected"] - comparison["mean_old_bad"]
    )
    comparison["relative_change"] = (
        comparison["absolute_change"] / comparison["mean_old_bad"].abs()
    )

    patch_monthly = pd.DataFrame(reports["patch"]["monthly_comparisons"])
    patch_2023 = (
        patch_monthly[patch_monthly["year"] == 2023]
        .groupby(["region", "variable"], observed=True)["patch_mean"]
        .mean()
        .reset_index()
    )
    corrected_2023 = corrected[corrected["year"] == 2023].copy()
    corrected_2023["variable"] = corrected_2023["variable"].map(VARIABLE_MAP)
    patch_check = corrected_2023.merge(
        patch_2023,
        on=["region", "variable"],
        validate="one_to_one",
    )
    patch_check["absolute_difference"] = (
        patch_check["mean"] - patch_check["patch_mean"]
    ).abs()
    patch_check["matches_patch"] = patch_check["absolute_difference"] <= 1e-5

    physical = reports["physical"]
    raw_dryness = physical["dryness_proxy"]
    dryness_log = {
        key: float(np.log1p(raw_dryness[key]))
        for key in ["p50", "p95", "p99", "max"]
    }
    blocking = []
    for key in ["raw", "processed"]:
        if reports[key]["status"] != "ready":
            blocking.append(f"{key} audit status={reports[key]['status']}")
    if physical["status"] != "ready":
        blocking.append(f"physical audit status={physical['status']}")
    if not patch_check["matches_patch"].all():
        blocking.append("Corrected 2023 accumulated means do not match patch")
    if corrected_rows != reports["processed"]["row_count"]:
        blocking.append("Corrected processed row count changed during audit")

    corrected_files = [
        {
            "path": str(Path(path)),
            "size_bytes": Path(path).stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in [
            *era5["input_files"],
            era5["processed_path"],
            config["features_path"],
        ]
    ]
    return {
        "status": "ready" if not blocking else "failed",
        "dataset_name": config["dataset_name"],
        "dataset_version": config["dataset_version"],
        "provenance": config["provenance"],
        "corrected_files": corrected_files,
        "raw_audit": reports["raw"],
        "processed_audit": reports["processed"],
        "physical_audit": physical,
        "corrected_processed_rows": corrected_rows,
        "old_processed_rows": old_rows,
        "negative_evaporation": {
            "old_bad_processed_count": old_negative,
            "corrected_processed_count": corrected_negative,
            "corrected_physical_count": physical[
                "negative_evaporation_count"
            ],
            "policy": physical["negative_evaporation_policy"],
        },
        "dryness_proxy_log1p": dryness_log,
        "old_bad_vs_corrected_yearly_means": comparison.to_dict("records"),
        "corrected_2023_vs_patch": patch_check.to_dict("records"),
        "blocking_issues": blocking,
    }


def write_corrected_full_audit(
    config: dict[str, Any], report: dict[str, Any]
) -> tuple[Path, Path]:
    json_path = Path(config["corrected_full_audit_json"])
    markdown_path = Path(config["corrected_full_audit_markdown"])
    for path in [json_path, markdown_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite corrected audit: {path}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    comparison = pd.DataFrame(report["old_bad_vs_corrected_yearly_means"])
    recent = comparison[comparison["year"].isin([2022, 2023])][
        [
            "region",
            "year",
            "variable",
            "mean_old_bad",
            "mean_corrected",
            "relative_change",
        ]
    ]
    files = pd.DataFrame(report["corrected_files"])
    text = f"""# Corrected ERA5-Land full-data audit

Status: **{report["status"]}**

Dataset version: `{report["dataset_version"]}`

## Artifacts

```csv
{files.to_csv(index=False).strip()}
```

## Old bad versus corrected accumulated yearly means

```csv
{recent.to_csv(index=False).strip()}
```

## Key checks

- Raw status: `{report["raw_audit"]["status"]}`.
- Processed status: `{report["processed_audit"]["status"]}`.
- Physical status: `{report["physical_audit"]["status"]}`.
- Corrected rows: `{report["corrected_processed_rows"]:,}`.
- Corrected negative evaporation count:
  `{report["negative_evaporation"]["corrected_processed_count"]}`.
- `dryness_proxy_log1p`: `{report["dryness_proxy_log1p"]}`.
- Corrected 2023 means match patch:
  `{all(row["matches_patch"] for row in report["corrected_2023_vs_patch"])}`.

Blocking issues: `{report["blocking_issues"]}`

No benchmark was run.
"""
    markdown_path.write_text(text, encoding="utf-8")
    return json_path, markdown_path
