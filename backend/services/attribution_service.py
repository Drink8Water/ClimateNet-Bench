"""Attribution and feature importance data access."""

from __future__ import annotations

import pandas as pd

from backend.config import get_experiment_dir, OUTPUTS_DIR, LEGACY_OUTPUTS_DIR
from backend.data_loader import read_csv_cached


def get_feature_importance(
    experiment_id: str,
    model: str | None = None,
    validation_strategy: str | None = None,
    region: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Return feature importance rows from an experiment or legacy outputs."""
    exp_dir = get_experiment_dir(experiment_id)
    path = exp_dir / "feature_importance.csv"

    # Fallback to legacy outputs if experiment doesn't have the file
    if not path.exists():
        path = LEGACY_OUTPUTS_DIR / "feature_importance.csv"
    if not path.exists():
        path = LEGACY_OUTPUTS_DIR / "feature_importance_by_region.csv"

    df = read_csv_cached(path)
    if df.empty:
        return []

    if model and "model" in df.columns:
        df = df[df["model"] == model]
    if validation_strategy and "validation_strategy" in df.columns:
        df = df[df["validation_strategy"] == validation_strategy]
    if region and "region" in df.columns:
        df = df[df["region"] == region]

    df = df.head(limit)
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def get_shap_info(experiment_id: str) -> dict:
    """Check if SHAP plots exist for an experiment."""
    from backend.config import PROJECT_ROOT

    exp_dir = get_experiment_dir(experiment_id)
    plot_paths = []

    def _rel_path(p: Path) -> str:
        """Convert absolute path to string relative to PROJECT_ROOT, or just the filename."""
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return p.name

    # Check experiment plots directory
    plots_dir = exp_dir / "plots"
    shap_patterns = ["shap", "regional_shap"]
    if plots_dir.exists():
        for pattern in shap_patterns:
            for p in sorted(plots_dir.glob(f"*{pattern}*")):
                plot_paths.append(_rel_path(p))

    # Also check legacy plots
    legacy_plots = LEGACY_OUTPUTS_DIR
    shap_patterns_legacy = ["shap_summary", "regional_shap"]
    if legacy_plots.exists():
        for pattern in shap_patterns_legacy:
            for p in sorted(legacy_plots.glob(f"*{pattern}*")):
                rel = _rel_path(p)
                if rel not in plot_paths:
                    plot_paths.append(rel)

    return {
        "available": len(plot_paths) > 0,
        "experiment_id": experiment_id,
        "plot_paths": plot_paths,
    }


def get_local_explanations(
    experiment_id: str,
    model: str | None = None,
    region: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Return feature importance as local explanation proxy (top features per grid point).

    Since the current pipeline doesn't produce per-sample SHAP values,
    this returns the top global features with their importance as a proxy.
    """
    importance = get_feature_importance(experiment_id, model=model, region=region, limit=limit)

    # Enrich with rank and contribution label
    results = []
    for i, row in enumerate(importance):
        imp = row.get("importance") or row.get("importance_mean") or 0
        feature = row.get("feature", "unknown")
        dominant = i == 0  # first row is dominant feature
        results.append({
            "grid_id": f"grid_{i:04d}",
            "region": row.get("region") or region or "all",
            "feature": feature,
            "shap_score": imp,
            "importance_type": row.get("importance_type", "global"),
            "dominant": dominant,
        })
    return results
