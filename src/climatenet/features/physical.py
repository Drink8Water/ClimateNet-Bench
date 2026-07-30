"""Physically informed climate feature engineering."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def saturation_vapor_pressure(temperature_celsius: pd.Series) -> pd.Series:
    """Calculate saturation vapor pressure in kPa from Celsius temperature."""
    return 0.6108 * np.exp((17.27 * temperature_celsius) / (temperature_celsius + 237.3))


def add_physical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add physical predictors used by ClimateNet models."""
    features = data.copy()
    features["wind_speed"] = np.sqrt(features["u_wind"] ** 2 + features["v_wind"] ** 2)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    features["dryness_proxy"] = features["radiation"] / (features["precipitation"] + 1e-6)
    # The raw ratio is retained for audit only. The formal v1 feature set uses
    # this row-wise bounded-dynamic-range transform, which needs no fitted
    # statistic and therefore cannot leak validation/test information.
    if (features["dryness_proxy"] < 0).any():
        raise ValueError(
            "dryness_proxy_log1p requires non-negative radiation/precipitation"
        )
    features["dryness_proxy_log1p"] = np.log1p(features["dryness_proxy"])
    features["saturation_vapor_pressure"] = saturation_vapor_pressure(features["temperature"])
    return features


def build_physical_features_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    audit_path: str | Path | None = None,
    chunksize: int = 250_000,
) -> dict[str, Any]:
    """Stream row-wise physical features to CSV and return an audit report.

    No climatology or anomaly is fitted here. Existing outputs and partial
    files are refused so a valid full-data artifact cannot be overwritten.
    """
    source = Path(input_path)
    output = Path(output_path)
    partial = output.with_suffix(output.suffix + ".partial")
    if not source.is_file():
        raise FileNotFoundError(f"Physical-feature input not found: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if partial.exists():
        raise FileExistsError(
            f"Partial output already exists; inspect before retrying: {partial}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    non_finite_counts: dict[str, int] = {}
    dryness_values: list[np.ndarray] = []
    svp_min = float("inf")
    svp_max = float("-inf")
    negative_evaporation_count = 0
    columns: list[str] = []
    try:
        for chunk_index, chunk in enumerate(
            pd.read_csv(source, chunksize=chunksize)
        ):
            transformed = add_physical_features(chunk)
            columns = transformed.columns.tolist()
            numeric = transformed.select_dtypes(include=[np.number])
            for column in numeric:
                count = int((~np.isfinite(numeric[column].to_numpy())).sum())
                non_finite_counts[column] = (
                    non_finite_counts.get(column, 0) + count
                )
            dryness_values.append(
                transformed["dryness_proxy"].to_numpy(dtype=np.float64)
            )
            svp = transformed["saturation_vapor_pressure"].to_numpy(
                dtype=np.float64
            )
            svp_min = min(svp_min, float(np.min(svp)))
            svp_max = max(svp_max, float(np.max(svp)))
            negative_evaporation_count += int(
                (transformed["evaporation"] < 0).sum()
            )
            transformed.to_csv(
                partial,
                mode="w" if chunk_index == 0 else "a",
                header=chunk_index == 0,
                index=False,
            )
            row_count += len(transformed)
        partial.replace(output)
    except Exception:
        # Preserve a partial artifact for diagnosis; never present it as final.
        raise

    dryness = np.concatenate(dryness_values)
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    report: dict[str, Any] = {
        "status": (
            "ready" if not any(non_finite_counts.values()) else "warning"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transformation_scope": "row_wise_only_no_climatology_or_anomaly_fit",
        "input_path": str(source.resolve()),
        "output_path": str(output.resolve()),
        "row_count": int(row_count),
        "columns": columns,
        "non_finite_counts": non_finite_counts,
        "dryness_proxy": {
            "role": "audit_only_not_used_directly_by_formal_v1",
            "p50": float(np.quantile(dryness, 0.50)),
            "p95": float(np.quantile(dryness, 0.95)),
            "p99": float(np.quantile(dryness, 0.99)),
            "max": float(np.max(dryness)),
        },
        "dryness_proxy_log1p": {
            "role": "formal_v1_row_wise_feature",
            "min": float(np.log1p(np.min(dryness))),
            "max": float(np.log1p(np.max(dryness))),
        },
        "saturation_vapor_pressure": {
            "min": svp_min,
            "max": svp_max,
        },
        "negative_evaporation_count": negative_evaporation_count,
        "negative_evaporation_policy": (
            "retained unchanged; may represent condensation/dew or ERA5 "
            "flux-sign/accumulation nuance"
        ),
        "size_bytes": output.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    if audit_path is not None:
        destination = Path(audit_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return report
