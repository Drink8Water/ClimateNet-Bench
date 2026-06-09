"""Physical consistency audit for ClimateNet-Bench.

Audits whether model predictions are broadly consistent with
atmospheric-science expectations.  This is a **model behaviour audit**,
NOT causal discovery.

Audit questions
---------------
1. Does increasing radiation_anomaly tend to increase predicted evaporation?
2. Does increasing soil_moisture_anomaly tend to increase predicted evaporation,
   especially in water-limited (arid/semi_arid) regions?
3. Does the model rely more on soil moisture in arid regions and more on
   precipitation / seasonality in monsoon regions?
4. Are regional attribution patterns physically plausible?

Each question is answered with:
- A **feature sensitivity curve** (vary one feature, hold others fixed).
- A **monotonic trend check** (Spearman ρ between feature value and response).
- **Regional breakdowns** where applicable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature registry — which features to audit and what sign we expect
# ---------------------------------------------------------------------------

AUDIT_FEATURES: dict[str, dict[str, str | list[str]]] = {
    "radiation_anomaly_lag_1": {
        "label": "Radiation Anomaly (lag_1)",
        "expected_sign": "positive",
        "rationale": "More surface solar radiation provides more energy for evaporation.",
    },
    "soil_moisture_anomaly_lag_1": {
        "label": "Soil Moisture Anomaly (lag_1)",
        "expected_sign": "positive",
        "rationale": "Wetter soil supplies more water for evaporation.  Effect should be strongest in water-limited (arid/semi-arid) regions.",
    },
    "temperature_anomaly_lag_1": {
        "label": "Temperature Anomaly (lag_1)",
        "expected_sign": "positive",
        "rationale": "Higher temperatures increase evaporative demand via the Clausius-Clapeyron relation.",
    },
    "precipitation_anomaly_lag_1": {
        "label": "Precipitation Anomaly (lag_1)",
        "expected_sign": "positive",
        "rationale": "More precipitation wets the surface, enabling more evaporation.",
    },
    "dryness_proxy_lag_1": {
        "label": "Dryness Proxy (lag_1)",
        "expected_sign": "negative",
        "rationale": "Higher dryness (radiation / precipitation) indicates water stress → less evaporation.",
    },
    "wind_speed_lag_1": {
        "label": "Wind Speed (lag_1)",
        "expected_sign": "positive",
        "rationale": "Stronger winds enhance turbulent moisture transport away from the surface.",
    },
}

# Features where the regional contrast is most diagnostic
REGIONALLY_SENSITIVE_FEATURES = [
    "soil_moisture_anomaly_lag_1",
    "precipitation_anomaly_lag_1",
    "radiation_anomaly_lag_1",
    "dryness_proxy_lag_1",
]

# ---------------------------------------------------------------------------
# Feature sensitivity curve
# ---------------------------------------------------------------------------


def compute_feature_sensitivity(
    model: Any,
    df: pd.DataFrame,
    feature: str,
    n_points: int = 20,
    quantile_range: tuple[float, float] = (0.05, 0.95),
    feature_cols: list[str] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Vary one feature while holding others at median; return the response curve.

    Parameters
    ----------
    model
        Any object with a ``predict(X)`` method returning 1-D predictions.
    df
        DataFrame containing feature columns.
    feature
        Column name to vary.
    n_points
        Number of points along the feature range.
    quantile_range
        (low, high) quantiles defining the variation range.
    feature_cols
        All feature columns the model expects.  If ``None``, uses
        ``df.columns``.
    seed
        Random seed for reproducibility.

    Returns
    -------
    Dict with keys ``feature_values``, ``mean_prediction``,
    ``std_prediction``, ``feature``.
    """
    rng = np.random.default_rng(seed)
    if feature_cols is None:
        feature_cols = list(df.columns)

    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not in DataFrame columns.")

    # Baseline: median values for all features
    median_row = df[feature_cols].median().to_frame().T
    n_baseline = min(200, len(df))
    baseline_samples = df[feature_cols].sample(n=n_baseline, random_state=seed)

    # Range to vary the target feature
    lo = float(df[feature].quantile(quantile_range[0]))
    hi = float(df[feature].quantile(quantile_range[1]))
    grid = np.linspace(lo, hi, n_points)

    means = np.zeros(n_points)
    stds = np.zeros(n_points)

    for i, val in enumerate(grid):
        perturbed = baseline_samples.copy()
        perturbed[feature] = val
        preds = model.predict(perturbed[feature_cols].to_numpy(dtype=np.float64))
        means[i] = float(np.mean(preds))
        stds[i] = float(np.std(preds))

    return {
        "feature": feature,
        "feature_values": grid,
        "mean_prediction": means,
        "std_prediction": stds,
    }


# ---------------------------------------------------------------------------
# Regional feature sensitivity
# ---------------------------------------------------------------------------


def compute_regional_sensitivity(
    model: Any,
    df: pd.DataFrame,
    feature: str,
    region_col: str = "region",
    n_points: int = 15,
    feature_cols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute feature sensitivity per region.

    Returns a DataFrame with columns ``region``, ``feature_value``,
    ``mean_prediction``, ``feature``.
    """
    if feature_cols is None:
        feature_cols = list(df.columns)

    rows: list[dict[str, Any]] = []
    for region_name, region_df in df.groupby(region_col):
        curve = compute_feature_sensitivity(
            model=model,
            df=region_df,
            feature=feature,
            n_points=n_points,
            feature_cols=feature_cols,
            seed=seed,
        )
        for fv, mp in zip(curve["feature_values"], curve["mean_prediction"]):
            rows.append(
                {
                    "region": region_name,
                    "feature": feature,
                    "feature_value": fv,
                    "mean_prediction": mp,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monotonic trend check
# ---------------------------------------------------------------------------


def check_monotonic_trend(
    feature_values: np.ndarray,
    mean_predictions: np.ndarray,
) -> dict[str, Any]:
    """Estimate whether the response curve is mostly increasing or decreasing.

    Uses Spearman rank correlation ρ.  ρ ≈ +1 → monotonic increasing;
    ρ ≈ −1 → monotonic decreasing; ρ ≈ 0 → no clear trend.

    Returns
    -------
    Dict with keys ``spearman_rho``, ``p_value``, ``direction``,
    ``is_significant`` (p < 0.05), ``monotonic`` (|ρ| > 0.7).
    """
    rho, p_val = spearmanr(feature_values, mean_predictions)

    direction = "increasing" if rho > 0 else "decreasing" if rho < 0 else "flat"

    return {
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "direction": direction,
        "is_significant": float(p_val) < 0.05,
        "monotonic": abs(rho) > 0.7,
    }


def is_consistent_with_expectation(
    trend: dict[str, Any],
    expected_sign: str,
) -> bool:
    """Check whether the monotonic trend matches the physically expected sign.

    Parameters
    ----------
    trend
        Output of ``check_monotonic_trend``.
    expected_sign
        ``"positive"`` or ``"negative"``.

    Returns
    -------
    ``True`` if the direction matches expectation AND is significant.
    """
    if expected_sign == "positive":
        return trend["direction"] == "increasing" and trend["is_significant"]
    elif expected_sign == "negative":
        return trend["direction"] == "decreasing" and trend["is_significant"]
    return False


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------


def run_physical_audit(
    model: Any,
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str | Path,
    region_col: str = "region",
    audit_features: dict | None = None,
) -> dict[str, Any]:
    """Run the complete physical consistency audit and save outputs.

    Parameters
    ----------
    model
        Trained model with a ``predict(X)`` method.
    model_name
        Human-readable model name for the report.
    df
        DataFrame with feature columns and ``region_col``.
    feature_cols
        Ordered list of feature columns the model expects.
    output_dir
        Directory for plots and CSV/JSON outputs.
    region_col
        Column identifying the region (default ``"region"``).
    audit_features
        Dict of ``feature_name → {label, expected_sign, rationale}``.
        Defaults to ``AUDIT_FEATURES``.

    Returns
    -------
    Dict with keys ``model_name``, ``consistency_score``,
    ``feature_results``, ``regional_results``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if audit_features is None:
        audit_features = AUDIT_FEATURES

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Only audit features that exist in the data AND in the feature list
    available = {
        f: v
        for f, v in audit_features.items()
        if f in df.columns and f in feature_cols
    }
    if not available:
        logger.warning("No audit features found in data. Skipping physical audit.")
        return {"model_name": model_name, "error": "No audit features in data."}

    feature_results: list[dict[str, Any]] = []
    regional_rows: list[dict[str, Any]] = []
    consistent_count = 0
    total_checks = 0

    # ── per-feature audit ──────────────────────────────────────────
    for feature, meta in available.items():
        curve = compute_feature_sensitivity(
            model, df, feature, feature_cols=feature_cols
        )
        trend = check_monotonic_trend(
            curve["feature_values"], curve["mean_prediction"]
        )

        expected_sign = str(meta.get("expected_sign", "positive"))
        is_consistent = is_consistent_with_expectation(trend, expected_sign)
        if is_consistent:
            consistent_count += 1
        total_checks += 1

        feature_results.append(
            {
                "feature": feature,
                "label": meta.get("label", feature),
                "expected_sign": expected_sign,
                "spearman_rho": trend["spearman_rho"],
                "p_value": trend["p_value"],
                "direction": trend["direction"],
                "monotonic": bool(trend["monotonic"]),
                "physically_consistent": bool(is_consistent),
                "rationale": meta.get("rationale", ""),
            }
        )

        # ── plot ───────────────────────────────────────────────
        _plot_sensitivity_curve(
            curve=curve,
            trend=trend,
            meta=meta,
            output_path=out / f"pdp_{feature}.png",
        )

        # ── regional for selected features ──────────────────────
        if feature in REGIONALLY_SENSITIVE_FEATURES and region_col in df.columns:
            reg_df = compute_regional_sensitivity(
                model, df, feature, region_col=region_col, feature_cols=feature_cols
            )
            regional_rows.append(reg_df)
            _plot_regional_sensitivity(
                reg_df=reg_df,
                feature=feature,
                meta=meta,
                output_path=out / f"pdp_{feature}_by_region.png",
            )

    # ── consistency score ───────────────────────────────────────
    consistency_score = consistent_count / total_checks if total_checks > 0 else 0.0

    # ── save artifacts ───────────────────────────────────────────
    summary = {
        "model_name": model_name,
        "consistency_score": consistency_score,
        "n_features_audited": total_checks,
        "n_physically_consistent": consistent_count,
        "feature_results": feature_results,
    }

    with (out / "consistency_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if regional_rows:
        regional_df = pd.concat(regional_rows, ignore_index=True)
        regional_df.to_csv(out / "regional_sensitivity.csv", index=False)
        summary["regional_sensitivity_path"] = str(out / "regional_sensitivity.csv")

    # ── markdown report ──────────────────────────────────────────
    report_path = out / "physical_consistency_report.md"
    _write_markdown_report(
        summary=summary,
        audit_features=audit_features,
        model_name=model_name,
        output_path=report_path,
    )
    summary["report_path"] = str(report_path)

    return summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_sensitivity_curve(
    curve: dict,
    trend: dict,
    meta: dict,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = curve["feature_values"]
    y = curve["mean_prediction"]
    std = curve["std_prediction"]

    ax.fill_between(x, y - std, y + std, alpha=0.25, color="steelblue")
    ax.plot(x, y, color="steelblue", linewidth=2)
    ax.set_xlabel(meta.get("label", curve["feature"]))
    ax.set_ylabel("Mean Predicted Evaporation Anomaly")
    ax.set_title(
        f"Sensitivity: {meta.get('label', curve['feature'])}\n"
        f"ρ = {trend['spearman_rho']:.3f}  |  "
        f"{trend['direction']}  |  "
        f"expected {meta.get('expected_sign', '?')}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _plot_regional_sensitivity(
    reg_df: pd.DataFrame,
    feature: str,
    meta: dict,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    regions = reg_df["region"].unique()
    n_regions = len(regions)
    if n_regions == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_regions, 1)))

    for i, region in enumerate(regions):
        sub = reg_df[reg_df["region"] == region]
        ax.plot(
            sub["feature_value"],
            sub["mean_prediction"],
            color=colors[i],
            linewidth=2,
            label=region,
        )

    ax.set_xlabel(meta.get("label", feature))
    ax.set_ylabel("Mean Predicted Evaporation Anomaly")
    ax.set_title(f"Regional Sensitivity: {meta.get('label', feature)}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------


def _write_markdown_report(
    summary: dict,
    audit_features: dict,
    model_name: str,
    output_path: Path,
) -> None:
    lines: list[str] = []
    w = lines.append

    w("# Physical Consistency Audit Report")
    w("")
    w(f"**Model:** `{model_name}`")
    w(f"**Consistency Score:** {summary['consistency_score']:.2f} "
      f"({summary['n_physically_consistent']}/{summary['n_features_audited']} "
      f"features consistent with physical expectations)")
    w("")
    w("---")
    w("")
    w("## Purpose")
    w("")
    w("This audit checks whether the model's learned relationships are "
      "broadly consistent with atmospheric-science expectations. "
      "It is a **model behaviour diagnostic**, NOT causal discovery.")
    w("")
    w("A model that passes these checks earns credibility: its predictions "
      "respond to input perturbations in ways that align with known physics. "
      "A model that fails may have learned spurious correlations from the "
      "training data.")
    w("")
    w("---")
    w("")
    w("## Feature-by-Feature Results")
    w("")

    for fr in summary["feature_results"]:
        status = "✅" if fr["physically_consistent"] else "⚠️"
        w(f"### {status} {fr['label']}")
        w("")
        w(f"- **Expected sign:** {fr['expected_sign']}")
        w(f"- **Observed direction:** {fr['direction']} "
          f"(ρ = {fr['spearman_rho']:.3f}, p = {fr['p_value']:.4f})")
        w(f"- **Monotonic:** {'yes' if fr['monotonic'] else 'no'}")
        w(f"- **Physically consistent:** {'yes' if fr['physically_consistent'] else '**no** — investigate'}")
        w(f"- **Rationale:** {fr['rationale']}")
        w("")
        w(f"![{fr['label']}](pdp_{fr['feature']}.png)")
        w("")

    w("---")
    w("")
    w("## Regional Findings")
    w("")
    w("Features with expected regional contrast were audited per region:")
    w("")
    for feat in REGIONALLY_SENSITIVE_FEATURES:
        if feat in audit_features:
            w(f"- **{audit_features[feat]['label']}** — "
              f"see `pdp_{feat}_by_region.png`")
    w("")

    w("---")
    w("")
    w("## Limitations")
    w("")
    w("1. **Not causal.**  Feature sensitivity curves show model *response*, "
      "not causal effects.  A model may increase its prediction when radiation "
      "increases because it learned a correlation, not because it understands "
      "the surface energy balance.")
    w("")
    w("2. **One-feature-at-a-time.**  Varying one feature while holding others "
      "fixed ignores interactions (e.g. radiation and soil moisture often "
      "co-vary in the real world).  Partial dependence plots partially address "
      "this by marginalising over the empirical distribution of other features.")
    w("")
    w("3. **No ground truth for evaporation.**  ERA5-Land evaporation is "
      "model-derived, not observed.  'Physically consistent' means consistent "
      "with the reanalysis model's representation of physics, not necessarily "
      "with real-world measurements.")
    w("")
    w("4. **Lag-1 features only.**  This audit only examines lag-1 features "
      "for simplicity.  Lag-2 through lag-6 features may show different "
      "behaviour.")
    w("")
    w("5. **Synthetic / demo data.**  If the model was trained on synthetic "
      "data, the audit results reflect the synthetic generator's assumptions, "
      "not real climate physics.")
    w("")
    w("---")
    w("")
    w("*Generated by ClimateNet-Bench physical consistency audit.*")

    output_path.write_text("\n".join(lines), encoding="utf-8")
