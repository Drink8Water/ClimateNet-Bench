"""Streamlit dashboard for local climate ML outputs and API records."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "latest"
EXPERIMENT_PLOTS_DIR = EXPERIMENT_DIR / "plots"

load_dotenv(PROJECT_ROOT / ".env")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def local_csv(path: Path) -> pd.DataFrame:
    """Read a local CSV with a clear dashboard error."""
    if not path.exists():
        st.error(f"找不到文件：{path}。请先运行对应的 pipeline 脚本。")
        return pd.DataFrame()
    return pd.read_csv(path)


def first_existing_path(paths: list[Path]) -> Path | None:
    """Return the first local path that exists."""
    for path in paths:
        if path.exists():
            return path
    return None


def local_csv_fallback(paths: list[Path]) -> pd.DataFrame:
    """Read the first available CSV from a list of fallback paths."""
    path = first_existing_path(paths)
    if path is None:
        checked = "\n".join(str(item) for item in paths)
        st.error(f"找不到任何可用 CSV。已检查：\n{checked}")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=30)
def api_get(endpoint: str, params: dict[str, object] | None = None) -> pd.DataFrame:
    """Read data from FastAPI and return an empty dataframe on failure."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=2)
        response.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame()
    return pd.DataFrame(response.json())


def get_metrics() -> tuple[pd.DataFrame, str]:
    """Load metrics from API first, then local CSV."""
    data = api_get("/metrics")
    if not data.empty:
        return data, "FastAPI"

    local = local_csv_fallback([EXPERIMENT_DIR / "metrics.csv", OUTPUTS_DIR / "all_metrics.csv"])
    local = local.rename(columns={"MAE": "mae", "RMSE": "rmse", "R2": "r2"})
    return local, "local CSV"


def get_predictions(limit: int = 2000) -> tuple[pd.DataFrame, str]:
    """Load predictions from API first, then local CSV."""
    data = api_get("/predictions", {"limit": limit})
    if not data.empty:
        return data, "FastAPI"
    return local_csv_fallback([EXPERIMENT_DIR / "predictions.csv", OUTPUTS_DIR / "predictions.csv"]).head(limit), "local CSV"


def get_feature_importance() -> tuple[pd.DataFrame, str]:
    """Load feature importance from API first, then local CSV."""
    data = api_get("/feature-importance", {"limit": 200})
    if not data.empty:
        return data, "FastAPI"
    return local_csv_fallback(
        [
            EXPERIMENT_DIR / "feature_importance.csv",
            OUTPUTS_DIR / "feature_importance_by_region.csv",
            OUTPUTS_DIR / "feature_importance.csv",
        ]
    ), "local CSV"


def get_regional_summary() -> tuple[pd.DataFrame, str]:
    """Load regional summary from API first, then calculate from local features."""
    data = api_get("/regional-summary")
    if not data.empty:
        return data, "FastAPI"

    features = local_csv_fallback([EXPERIMENT_DIR / "features.csv", PROCESSED_DIR / "features.csv"])
    if features.empty:
        return features, "local CSV"

    summary = (
        features.groupby("region")
        .agg(
            n_records=("region", "size"),
            mean_temperature=("temperature", "mean"),
            mean_precipitation=("precipitation", "mean"),
            mean_radiation=("radiation", "mean"),
            mean_soil_moisture=("soil_moisture", "mean"),
            mean_evaporation_anomaly=("evaporation_anomaly", "mean"),
        )
        .reset_index()
    )
    return summary, "local CSV"


def show_source_badge(source: str) -> None:
    """Show whether the dashboard is using API or local files."""
    st.caption(f"数据来源：{source}")


def page_overview() -> None:
    """Project overview page."""
    st.header("Project overview")
    st.write(
        "这个 dashboard 展示气候 ML 管线的已保存输出：特征表、模型指标、预测结果、"
        "特征重要性和 SHAP 解释图。它优先读取 FastAPI；如果后端没启动，则自动读取本地 CSV/PNG。"
    )

    metrics, metric_source = get_metrics()
    summary, summary_source = get_regional_summary()

    col1, col2, col3 = st.columns(3)
    col1.metric("模型评估行数", len(metrics))
    col2.metric("区域数量", summary["region"].nunique() if not summary.empty else 0)
    col3.metric("验证策略数量", metrics["validation_strategy"].nunique() if not metrics.empty else 0)

    show_source_badge(f"metrics={metric_source}, regional_summary={summary_source}")
    if not summary.empty:
        st.subheader("Regional summary")
        st.dataframe(summary, use_container_width=True)


def page_metrics() -> None:
    """Model metrics page."""
    st.header("Model metrics")
    metrics, source = get_metrics()
    show_source_badge(source)
    if metrics.empty:
        return

    st.dataframe(metrics, use_container_width=True)
    fig = px.bar(
        metrics,
        x="validation_strategy",
        y="rmse",
        color="model",
        barmode="group",
        hover_data=["train_region", "test_region", "mae", "r2"],
        title="RMSE by validation strategy",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_predictions() -> None:
    """Prediction vs actual page."""
    st.header("Prediction vs actual")
    predictions, source = get_predictions()
    show_source_badge(source)
    if predictions.empty:
        return

    models = sorted(predictions["model"].dropna().unique())
    strategies = sorted(predictions["validation_strategy"].dropna().unique())
    model = st.selectbox("Model", models)
    strategy = st.selectbox("Validation strategy", strategies)
    plot_data = predictions[
        (predictions["model"] == model)
        & (predictions["validation_strategy"] == strategy)
    ]

    fig = px.scatter(
        plot_data,
        x="actual",
        y="prediction",
        color="region",
        opacity=0.55,
        title=f"{model} - {strategy}",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(plot_data.head(200), use_container_width=True)


def page_feature_importance() -> None:
    """Feature importance page."""
    st.header("Feature importance")
    importance, source = get_feature_importance()
    show_source_badge(source)
    if importance.empty:
        return

    value_column = "importance_mean" if "importance_mean" in importance.columns else "importance"
    plot_data = importance.dropna(subset=[value_column]).copy()
    if "region" in plot_data.columns and plot_data["region"].notna().any():
        fig = px.bar(
            plot_data.sort_values(value_column, ascending=False).head(30),
            x=value_column,
            y="feature",
            color="region",
            orientation="h",
            title="Regional feature importance",
        )
    else:
        fig = px.bar(
            plot_data.sort_values(value_column, ascending=False).head(20),
            x=value_column,
            y="feature",
            color="model" if "model" in plot_data.columns else None,
            orientation="h",
            title="Global feature importance",
        )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(importance, use_container_width=True)


def page_regional_comparison() -> None:
    """Regional comparison page."""
    st.header("Regional comparison")
    summary, source = get_regional_summary()
    show_source_badge(source)
    if summary.empty:
        return

    st.dataframe(summary, use_container_width=True)
    numeric_columns = [column for column in summary.columns if column.startswith("mean_")]
    metric = st.selectbox("Regional variable", numeric_columns)
    fig = px.bar(summary, x="region", y=metric, color="region", title=metric)
    st.plotly_chart(fig, use_container_width=True)


def page_shap_images() -> None:
    """SHAP explanation image page."""
    st.header("SHAP explanation images")
    image_paths = [
        EXPERIMENT_PLOTS_DIR / "shap_summary.png",
        EXPERIMENT_PLOTS_DIR / "regional_shap_sahara.png",
        EXPERIMENT_PLOTS_DIR / "regional_shap_east_china.png",
    ]

    for path in image_paths:
        st.subheader(path.stem)
        fallback = first_existing_path([path, OUTPUTS_DIR / path.name])
        if fallback is not None:
            st.image(str(fallback), use_container_width=True)
        else:
            st.warning(f"找不到 {path.name}。请运行：python src/explain.py")


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Climate ML Dashboard", layout="wide")
    st.title("Climate ML Analysis Dashboard")

    page = st.sidebar.radio(
        "Page",
        [
            "Project overview",
            "Model metrics",
            "Prediction vs actual",
            "Feature importance",
            "Regional comparison",
            "SHAP explanation images",
        ],
    )

    if page == "Project overview":
        page_overview()
    elif page == "Model metrics":
        page_metrics()
    elif page == "Prediction vs actual":
        page_predictions()
    elif page == "Feature importance":
        page_feature_importance()
    elif page == "Regional comparison":
        page_regional_comparison()
    else:
        page_shap_images()


if __name__ == "__main__":
    main()
