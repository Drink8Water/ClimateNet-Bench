# ClimateNet-Bench 架构审计报告

> **审计日期**: 2026-06-18  
> **审计范围**: 全部源码、配置、测试、文档  
> **项目版本**: v0.3.0  
> **目标升级**: 陆面水文气候极端事件评估基准 + 全栈评估平台  

---

## 目录

1. [总体评估](#1-总体评估)
2. [数据管线](#2-数据管线)
3. [预处理与异常计算](#3-预处理与异常计算)
4. [划分协议](#4-划分协议)
5. [模型层](#5-模型层)
6. [评估指标](#6-评估指标)
7. [后端 API](#7-后端-api)
8. [前端](#8-前端)
9. [测试](#9-测试)
10. [文档](#10-文档)
11. [基础设施与运维](#11-基础设施与运维)
12. [保留/重构/删除 决策](#12-保留重构删除-决策)

---

## 1. 总体评估

### 1.1 项目概况

ClimateNet-Bench 是一个面向陆地蒸发异常预测的时空机器学习基准测试框架。当前版本 (v0.3.0) 包含：

- **源语言**: Python 3.11（核心 ~7900 行）、JavaScript（前端 ~1500 行）
- **架构模式**: 双轨制——旧版脚本 (`src/*.py`, ~1300 行) 与新版包 (`src/climatenet/`, ~6600 行) 并存
- **数据存储**: 本地 CSV/JSON 文件为主，PostgreSQL 为辅（可选）
- **部署方式**: 手动启动 FastAPI + Vite 开发服务器，无容器化

### 1.2 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (Vue 3 + Vite)                    │
│          http://localhost:5173  —  11 个视图页面           │
│            axios → /api → proxy → :8000                  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│               后端 API (FastAPI, 34+ 端点)                │
│         routers/ (10个路由) + services/ (9个服务)          │
│         优先读取 CSV/JSON 文件; PostgreSQL 可选             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│         核心包 climatenet (src/climatenet/)               │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │ benchmark│   data   │evaluation│     models       │  │
│  │(区域注册 │(预测数据 │(指标/技能│  (ABC基类/工厂/   │  │
│  │ 6种划分) │ 集构建)  │分/共形/物│   7个模型实现)    │  │
│  │          │          │理审计)   │                  │  │
│  └──────────┴──────────┴──────────┴──────────────────┘  │
│  ┌──────────┬──────────┬──────────────────────────────┐  │
│  │ training │ features │      utils (config/paths)    │  │
│  │(基准运行 │(物理特征 │                              │  │
│  │ 器/实验) │ /异常)   │                              │  │
│  └──────────┴──────────┴──────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│          旧版脚本 (src/*.py) — 部分功能重复               │
│  config.py / train.py / features.py / validation.py     │
│  evaluate.py / explain.py / plot_results.py ...         │
└─────────────────────────────────────────────────────────┘
```

### 1.3 关键架构问题

| 问题 | 严重程度 | 描述 |
|------|---------|------|
| 双轨代码 | 🔴 高 | `src/` 旧版代码与 `src/climatenet/` 新版包共存，函数重复（如 `calculate_metrics` 出现 4 次） |
| 后端启动崩溃 | 🔴 高 | `backend/database.py` 在无 DATABASE_URL 时抛出 `RuntimeError`，阻止整个 FastAPI 应用启动 |
| 无容器化 | 🟡 中 | 项目依赖 conda + pip 手动安装，无 Dockerfile 或 docker-compose |
| 前端无 TypeScript | 🟡 中 | 用户需求明确要求 TypeScript，当前为纯 JavaScript |
| 实验串行运行 | 🟡 中 | `benchmark_runner.py` 中 `(model × split × feature_set)` 实验完全串行 |
| outputs 入仓 | 🟡 中 | `outputs/` 目录 28MB 被 git 追踪 |
| 无异步任务 | 🟡 中 | 缺少 Redis/Celery 做长时间运行的基准测试任务 |
| 无对象存储 | 🟡 中 | 缺少 MinIO 做大型数据集和模型产物存储 |
| 无代码质量工具 | 🟢 低 | 无 ruff/mypy/pre-commit 配置 |

---

## 2. 数据管线

### 2.1 现有数据流

```
ERA5-Land CDS API
    │
    ▼
download_era5.py / climatenet/data/era5_download.py
    │  NetCDF (.nc) → data/raw/era5_land/
    ▼
preprocess_era5.py / climatenet/data/era5_preprocess.py
    │  xarray: 月度聚合, 区域子集, 基础物理量
    ▼
features.py / climatenet/features/pipeline.py
    │  物理特征: wind_speed, dryness_proxy, saturation_vapor_pressure
    │  异常值: 区域×月份气候学偏差
    │  时间编码: month_sin, month_cos
    ▼
data/processed/features.csv  (~数十万行)
    │
    ▼
build_forecasting_dataset.py / climatenet/data/forecasting_dataset.py
    │  滑动窗口 (6个月 lag) → 每个网格单元的 (sample_id, y_true, lag_1..lag_6)
    │  反泄漏保证: input_window_end < target_month
    ▼
data/processed/forecasting_samples.csv  (扁平表 + 3D序列数组)
```

### 2.2 数据管线审计

| 组件 | 文件 | 状态 | 评价 |
|------|------|------|------|
| ERA5 下载 | `src/download_era5.py`, `src/climatenet/data/era5_download.py` | 重复 | 两处实现，新包版本更规范 |
| NetCDF 预处理 | `src/preprocess_era5.py`, `src/climatenet/data/era5_preprocess.py` | 重复 | 同上 |
| 物理特征工程 | `src/physical_features.py`, `src/climatenet/features/physical.py` | 重复 | 旧版含 `add_physical_features()`，新版含同功能函数 |
| 异常值计算 | `src/features.py:add_climatology_and_anomalies()`, `src/climatenet/features/anomalies.py` | 重复 | 逻辑相同: `groupby(["region","month"]).transform("mean")` |
| 预测样本构建 | `src/climatenet/data/forecasting_dataset.py` | ✅ 良好 | 核心函数 `build_forecasting_samples()` — 含完整的反泄漏验证 |
| 3D序列数组 | `src/climatenet/data/sequence_dataset.py` | ✅ 良好 | TCN 模型专用 |
| 合成数据生成 | `src/make_sample_data.py`, `scripts/build_forecasting_dataset.py:_build_synthetic_features()` | 重复 | 用于冒烟测试 |

### 2.3 关键发现

**性能瓶颈** (P0):
- `build_forecasting_samples()` 使用 `for (region, lat, lon), group in data.groupby(...)` 嵌套循环，每个网格单元串行处理。对于 ERA5-Land 0.1° 分辨率的 5 个区域，预计有数万个网格单元——完全串行会很慢。
- 可以向量化: 使用 `shift()` + `rolling()` 操作替代逐行窗口构建。

**数据局限** (当前为 v0.3.0):
- 仅支持 ERA5-Land 数据源，无法接入其他再分析数据（ERA5、MERRA-2、JRA-55）
- 仅支持陆地蒸发异常目标变量，无法扩展到极端事件（干旱、洪水、热浪）
- 分辨率固定为 0.1° × 0.1°月度

---

## 3. 预处理与异常计算

### 3.1 现有特征清单

| 类别 | 特征 | 计算方式 |
|------|------|---------|
| **原始变量** | temperature, precipitation, radiation, soil_moisture, u_wind, v_wind, evaporation | 直接来自 ERA5-Land |
| **物理衍生** | wind_speed | `√(u² + v²)` |
| **物理衍生** | saturation_vapor_pressure | Clausius-Clapeyron: `0.6108 × exp(17.27×T/(T+237.3))` |
| **物理衍生** | dryness_proxy | `radiation / (precipitation + 1e-6)` |
| **异常值** | temperature_anomaly, precipitation_anomaly, radiation_anomaly, soil_moisture_anomaly, evaporation_anomaly | `value - climatology` (按 region×month 分组) |
| **时间编码** | month_sin, month_cos | `sin/cos(2π × month / 12)` |
| **空间** | latitude, longitude | 连续坐标 |
| **滞后值** | {feature}_lag_{1..6} | 由 `build_forecasting_samples()` 构建 |

### 3.2 审计评价

| 方面 | 状态 | 说明 |
|------|------|------|
| 气候学计算 | ✅ 正确 | 严格使用训练集分组均值，无泄漏 |
| 异常值推导 | ✅ 正确 | 减法操作，单位保持 |
| 物理一致性 | ✅ 良好 | Clausius-Clapeyron 公式正确；dryness_proxy 是合理的一阶近似 |
| 反泄漏 | ✅ 严格 | `input_window_end < target_month` 验证；滞后目标列仅用于 persistence baseline |
| **缺失特征** | 🔴 需补充 | 极端事件指标: SPI/SPEI、百分位阈值、连续干旱天数、极端降水指数 |

**升级需求**: 对于陆面水文气候极端事件评估，需要新增:
- **干旱指标**: SPI (标准化降水指数)、SPEI、scPDSI
- **极端降水**: Rx5day (连续5天最大降水)、R95p (95百分位降水)
- **热浪**: 热浪持续时间、极端高温日数
- **土壤水胁迫**: 土壤水分百分位、植物可用水

---

## 4. 划分协议

### 4.1 现有 6 种协议

| # | 协议 | 实现文件 | 拆分维度 | 测试覆盖 |
|---|------|---------|---------|---------|
| 1 | random | `split_protocols.py:make_random_split()` | 样本级随机 | ✅ 7个测试 |
| 2 | spatial_block | `split_protocols.py:make_spatial_block_split()` | 网格单元, 按 block_size_deg 分块 | ✅ 5个测试 |
| 3 | temporal | `split_protocols.py:make_temporal_split()` | 年份 (train_years / val_year / test_year) | ✅ 5个测试 |
| 4 | region_transfer | `split_protocols.py:make_region_transfer_split()` | 区域标签 | ✅ 5个测试 |
| 5 | climate_zone_transfer | `split_protocols.py:make_climate_zone_transfer_split()` | 气候分类 | ✅ 4个测试 |
| 6 | spatiotemporal | `split_protocols.py:make_spatiotemporal_split()` | 空间块 × 时间年份 | ✅ 3个测试 |

### 4.2 审计评价

| 方面 | 状态 | 说明 |
|------|------|------|
| API 设计 | ✅ 优秀 | 基于 sample_id 的拆分，数据帧零拷贝共享 |
| 反泄漏验证 | ✅ 严格 | `validate_split()` 检查 ID 重叠、空间泄漏、时间泄漏 |
| 确定性 | ✅ 可复现 | 所有拆分支持 seed 参数 |
| I/O | ✅ 完整 | `save/load_split_result()` 含 CSV + JSON metadata |
| 旧版重复 | 🟡 | `src/validation.py` 含较简单的 `random_split/spatial_holdout/cross_region_transfer` |

**升级需求**:
- 当前 `region_transfer` 和 `climate_zone_transfer` 生成所有可能的 train→test 对。对于 5 个区域 = 20 个方向。需要支持子集配置。
- 缺少 **极端事件分层划分**: 确保 train/test 中都包含极端事件样本（按百分位分层）
- 缺少 **时间序列交叉验证**: 仅支持单次 temporal split，不支持滚动窗口 CV

---

## 5. 模型层

### 5.1 现有模型

| 模型 | 实现文件 | 类型 | 状态 |
|------|---------|------|------|
| ClimatologyBaseline | `models/climatology.py` | 基线 | ✅ 完整 (region_monthly / global_monthly) |
| PersistenceBaseline | `models/persistence.py` | 基线 | ✅ 完整 (ŷ_t = y_{t-1}) |
| LinearRegressionModel | `models/linear.py` | 线性 | ✅ 完整 (Ridge) |
| RandomForestModel | `models/tree_models.py` | 树集成 | ✅ 完整 |
| XGBoostModel | `models/tree_models.py` | 梯度提升 | ✅ 完整 (含 early stopping) |
| LightGBMModel | `models/tree_models.py` | 梯度提升 | ✅ 完整 (可选导入) |
| TCNRegressor | `models/tcn.py` | 深度学习 | ✅ 完整 (PyTorch, ~117行) |
| MockPhysicallyPlausibleModel | `tests/test_physical_consistency.py` | 测试用 | ✅ 仅测试 |

### 5.2 模型工厂

| 组件 | 文件 | 状态 |
|------|------|------|
| ClimateModel (ABC) | `models/base.py` | ✅ 清晰的 fit/predict/get_model_name 接口 |
| create_model() | `models/model_factory.py` | ✅ 名称→类的映射，支持 kwargs 传递 |
| 旧版 build_models() | `src/train.py:43-77` | 🔴 重复 — 直接实例化 sklearn 对象，不经过工厂 |
| 旧版 factory | `models/factory.py:build_models()` | 🟡 另一个 YAML 驱动的工厂，与 model_factory.py 功能重叠 |

### 5.3 审计评价

| 方面 | 状态 | 说明 |
|------|------|------|
| 基类设计 | ✅ 良好 | fit/predict/save/load 接口清晰 |
| 序列化 | 🟡 可用 | 默认使用 pickle — 不适合生产环境长期存储 |
| 超参数管理 | 🟡 分散 | 散布在 YAML configs、train.py 硬编码、factory.py kwargs |
| GPU 支持 | 🟡 部分 | TCN 检测 CUDA，但树模型不利用 GPU（XGBoost/LightGBM 支持 GPU） |
| **缺失模型** | 🔴 需补充 | LSTM, Transformer, Graph Neural Network, 概率模型 (Gaussian Process, Quantile Regression) |

---

## 6. 评估指标

### 6.1 现有指标

| 类别 | 指标 | 实现文件 | 测试 |
|------|------|---------|------|
| **基础回归** | MAE, RMSE, R² | `evaluation/metrics.py` | ✅ 10个测试 |
| **技能分数** | Skill vs Climatology, Skill vs Persistence | `evaluation/skill_score.py` | ✅ 11个测试 |
| **OOD 退化** | ΔRMSE = RMSE_strict − RMSE_random | `evaluation/ood_degradation.py` | ✅ 7个测试 |
| **共形预测** | 分位数拟合, 覆盖率, 间隔宽度, 分组评估 | `evaluation/conformal.py` + `calibration.py` | ✅ 20+个测试 |
| **物理一致性** | 特征敏感性 (PDP), 单调趋势 (Spearman), 区域分解 | `evaluation/physical_consistency.py` | ✅ 13个测试 |

### 6.2 审计评价

| 方面 | 状态 | 说明 |
|------|------|------|
| 输入验证 | ✅ 严格 | `_validate_inputs()` 检查 NaN、维度、空数组、长度不匹配 |
| 共形预测 | ✅ 完整 | 分体共形 (split-conformal)，支持分组评估 |
| 物理审计 | ✅ 创新 | PDP + Spearman 趋势 + 物理预期对照表 |
| **缺失指标** | 🔴 需补充 | **极端事件专用指标**: POD (检测概率), FAR (误报率), CSI (临界成功指数), ETS |
| **缺失指标** | 🔴 需补充 | **概率预报指标**: CRPS (连续排名概率分数), Brier Score, 可靠性图 |
| **缺失指标** | 🔴 需补充 | **空间指标**: RMSE 的空间分布图, 小波分解, 谱分析 |

---

## 7. 后端 API

### 7.1 FastAPI 架构

```
backend/main.py              ← 应用入口，CORS 中间件
backend/config.py            ← 路径常量、区域/模型/策略枚举
backend/schemas.py           ← 16 个 Pydantic 响应模型 + 旧版别名
backend/data_loader.py       ← 通用 CSV/JSON/YAML 读取工具
backend/database.py          ← PostgreSQL 连接 (有问题 — 见下文)
backend/crud.py              ← 只读 SQL 查询

backend/routers/ (10个路由模块):
├── benchmark.py             ← /benchmark/summary, /task, /regions, /splits
├── leaderboard.py           ← /leaderboard, /split-difficulty, /ablation-study
├── experiments.py           ← /experiments, /experiments/{id}
├── predictions.py           ← /experiments/{id}/predictions, /residuals, /prediction-summary
├── uncertainty.py           ← /uncertainty/calibration, /experiments/{id}/intervals
├── physical.py              ← /physical-consistency/summary, /regional-sensitivity
├── spatial.py               ← /spatial-grid, /timeseries, /grid-cell-detail
├── attribution.py           ← /experiments/{id}/feature-importance, /shap, /local-explanations
├── comparison.py            ← /model-comparison, /ablation-study
└── summary.py               ← /project-summary, /dataset-summary

backend/services/ (9个服务模块):
├── benchmark_service.py
├── leaderboard_service.py
├── experiment_service.py
├── prediction_service.py
├── uncertainty_service.py
├── physical_service.py
├── spatial_service.py
├── attribution_service.py
└── comparison_service.py
```

### 7.2 关键问题

| 问题 | 文件 | 严重程度 | 描述 |
|------|------|---------|------|
| **database.py 崩溃启动** | `backend/database.py:15-16` | 🔴 致命 | `if not DATABASE_URL: raise RuntimeError(...)` — 在 `main.py` 导入时即触发，即使数据库端点未被调用。`main.py:63-68` 有 try/except 但无法阻止 `database.py` 模块级的 `load_dotenv()` + 检查。 |
| **路由 `__init__.py` 缺失** | `backend/routers/` | 🟡 | 无 `__init__.py`，`main.py` 可以导入但不符合 Python 包规范 |
| **服务层过薄** | `backend/services/` | 🟢 低 | 大部分服务仅做 CSV 读取 + 过滤，无缓存策略、分页或查询优化 |
| **硬编码路径** | `backend/config.py` | 🟡 | 路径相对于 `PROJECT_ROOT`，容器化部署时不灵活 |
| **无认证/授权** | 全局 | 🟡 | API 完全开放，无任何认证机制 |
| **无速率限制** | 全局 | 🟢 低 | 无请求速率限制 |

### 7.3 PostgreSQL 集成状态

- `sql/schema.sql` 定义了 4 张表 (`climate_features`, `model_predictions`, `feature_importance`, `model_metrics`)
- `src/load_to_db.py` 负责从 CSV 导入到 PostgreSQL
- `backend/crud.py` 有 5 个只读查询函数
- 但这些仅在 `DATABASE_URL` 环境变量设置时才激活
- **当前事实**: 整个后端运行在文件模式，数据库功能实际上不可用

---

## 8. 前端

### 8.1 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 框架 | Vue 3 (Composition API) | ^3.5.34 |
| 构建 | Vite | ^8.0.12 |
| UI | Tailwind CSS | ^4.3.0 |
| 图表 | ECharts (via vue-echarts) | ^6.1.0 |
| HTTP | axios | ^1.17.0 |
| 路由 | vue-router | ^4.6.4 |
| 语言 | **JavaScript** (不是 TypeScript!) | — |

### 8.2 视图清单 (11个页面)

| 路由 | 视图组件 | 描述 | 数据端点 |
|------|---------|------|---------|
| `/` | Overview.vue | 基准概览、KPI | /benchmark/summary, /benchmark/task |
| `/leaderboard` | Leaderboard.vue | 排名结果表 | /leaderboard |
| `/split-difficulty` | SplitDifficulty.vue | RMSE 按划分协议柱状图 | /split-difficulty |
| `/forecast` | ForecastExplorer.vue | 预测 vs 实际散点图 + 时间序列 | /experiments, /predictions |
| `/uncertainty` | UncertaintyCalibration.vue | 覆盖率 vs 间隔宽度校准 | /uncertainty/calibration |
| `/physical` | PhysicalAudit.vue | 物理一致性分数 + 区域敏感性 | /physical-consistency/* |
| `/spatial` | SpatialDiagnostics.vue | 网格单元时间序列 | /spatial-grid, /timeseries |
| `/experiments` | (重定向到 /forecast) | 旧版兼容 | — |
| `/comparison` | (重定向到 /leaderboard) | 旧版兼容 | — |
| `/predictions` | (重定向到 /forecast) | 旧版兼容 | — |
| `/attribution` | (重定向到 /physical) | 旧版兼容 | — |

### 8.3 审计评价

| 方面 | 状态 | 说明 |
|------|------|------|
| 组件结构 | ✅ 良好 | common/ + layout/ + views/ 分层清晰 |
| 路由设计 | ✅ 合理 | 7个活跃页面 + 4个旧版重定向 |
| API 封装 | ✅ 清晰 | `api/climateApi.js` 统一管理所有端点 |
| TypeScript | 🔴 缺失 | 用户需求明确要求 TypeScript — 整个前端需迁移 |
| 错误处理 | 🟡 部分 | 仅有基本 try/catch，无全局错误拦截器 |
| 加载状态 | 🟡 部分 | 缺少统一的 loading/skeleton 组件 |
| 响应式设计 | 🟡 未知 | 有 Tailwind 但未验证移动端适配 |
| 测试 | 🔴 缺失 | **前端无任何测试** (.spec 或 .test 文件) |
| 状态管理 | 🟢 适度 | 无 Pinia/Vuex — 每个视图独立 fetch，对当前规模可接受 |

---

## 9. 测试

### 9.1 现有测试文件

| 测试文件 | 测试类/函数数 | 覆盖模块 |
|---------|-------------|---------|
| `test_metrics.py` | ~20个测试, 3个类 | evaluation/metrics, skill_score, ood_degradation |
| `test_validation.py` | 1个函数 | training/validation (spatial_holdout) |
| `test_split_protocols.py` | ~24个测试, 10个类 | benchmark/split_protocols — **最全面的测试文件** |
| `test_baselines.py` | ~19个测试, 7个类 | models (所有基线 + 工厂) |
| `test_conformal.py` | ~20个测试, 9个类 | evaluation/conformal + calibration |
| `test_physical_consistency.py` | ~14个测试, 7个类 | evaluation/physical_consistency |
| `test_feature_pipeline.py` | 1个函数 | features/pipeline |
| `test_forecasting_dataset.py` | ~18个测试, 5个类 | data/forecasting_dataset — **最全面的集成测试** |
| `test_sequence_dataset.py` | 1个函数 | data/sequence_dataset |
| `test_leaderboard.py` | 未知 | benchmark/leaderboard |
| `test_region_registry.py` | 未知 | benchmark/region_registry |

### 9.2 测试覆盖缺口

| 缺失范围 | 严重程度 | 说明 |
|---------|---------|------|
| **后端 API 测试** | 🔴 | 无任何 FastAPI TestClient 测试 |
| **前端测试** | 🔴 | 无任何 .spec 或 .test 文件 |
| **集成测试** | 🟡 | 无端到端训练→评估→API 测试 |
| **TCN 模型测试** | 🟡 | test_baselines 中 TCN 被标记为 "requires 3D" 但无独立测试 |
| **边界情况** | 🟢 | 现有测试覆盖了大部分边界 (NaN、空数组、长度不匹配) |

---

## 10. 文档

### 10.1 现有文档

| 文档 | 质量 | 说明 |
|------|------|------|
| README.md | ✅ 优秀 | 8.9KB, 中英文混排, 含快速开始、任务定义、项目结构 |
| CHANGELOG.md | ✅ 良好 | 三版本历史 (v0.1 → v0.3), Roadmap 清单 |
| CONTRIBUTING.md | ✅ 良好 | 安装说明、编码风格、如何贡献 |
| CITATION.cff | ✅ 规范 | 结构化引用元数据 |
| docs/task_definition.md | ✅ 完整 | 问题陈述、输入窗口、特征表、科学意义 |
| docs/benchmark_protocol.md | ✅ 完整 | 6种划分协议、7个模型、指标、反泄漏规则 |
| docs/reproduce.md | 待检查 | 复现步骤 |
| docs/api.md | 待检查 | 34个端点文档 |
| docs/uncertainty.md | 待检查 | 共形预测方法论 |
| docs/physical_consistency.md | 待检查 | 物理审计方法论 |
| docs/limitations.md | 待检查 | 已知限制 |
| docs/development_history.md | 待检查 | 开发历史档案 |
| .env.example | ✅ 最小 | 仅 DATABASE_URL 示例 |

### 10.2 缺失文档

- 🔴 **部署指南** — 无 Docker / 生产环境部署文档
- 🔴 **数据字典** — 无特征/列的详细文档
- 🟡 **模型卡片** — HuggingFace 风格的模型性能卡片

---

## 11. 基础设施与运维

### 11.1 CI/CD

| 方面 | 当前状态 | 问题 |
|------|---------|------|
| CI 平台 | GitHub Actions | `.github/workflows/ci.yml` |
| Python 版本 | 仅 3.11 | 无多版本矩阵 |
| 操作系统 | ubuntu-latest | 无 macOS/Windows |
| 缓存 | 无 | pip 依赖每次重新安装 |
| 代码覆盖率 | 无 | 未集成 coverage/pytest-cov |
| 代码检查 | 无 | 无 ruff/mypy 步骤 |

### 11.2 缺失的基础设施

| 组件 | 状态 | 升级需求 |
|------|------|---------|
| Docker | 🔴 无 | 需 Dockerfile + docker-compose.yml |
| PostgreSQL | 🟡 部分 (schema存在但未集成) | 需真正集成: 连接池 (asyncpg)、迁移 (Alembic) |
| Redis | 🔴 无 | 需用于 Celery 任务队列 + 缓存 |
| Celery | 🔴 无 | 需用于异步基准测试运行 |
| MinIO / S3 | 🔴 无 | 需用于大型数据集和模型产物存储 |
| Nginx | 🔴 无 | 需用于生产环境反向代理 |
| 监控 | 🔴 无 | 需 Prometheus + Grafana 或至少 healthcheck |
| 日志 | 🟡 部分 | 部分使用 logging，部分使用 print() — 需统一为 structlog |

---

## 12. 保留/重构/删除 决策

### 12.1 ✅ 保留 (核心资产)

| 路径/组件 | 原因 |
|-----------|------|
| `src/climatenet/benchmark/region_registry.py` | 设计良好，可直接扩展新区域 |
| `src/climatenet/benchmark/split_protocols.py` | 6 种协议实现完整，测试充分 |
| `src/climatenet/data/forecasting_dataset.py` | 核心数据构造函数，反泄漏保证严格 |
| `src/climatenet/evaluation/metrics.py` | 基础指标，输入验证完善 |
| `src/climatenet/evaluation/conformal.py` | 共形预测实现完整 |
| `src/climatenet/evaluation/physical_consistency.py` | 创新性物理审计 |
| `src/climatenet/models/base.py` (ClimateModel ABC) | 清晰的模型接口 |
| `src/climatenet/models/model_factory.py` | 工厂模式，可扩展 |
| `src/climatenet/models/climatology.py`, `persistence.py` | 基线模型 |
| `tests/` | 198+ 测试，覆盖良好 |
| `docs/` | 8 篇质量文档 |
| `sql/schema.sql` | PostgreSQL schema，升级后可用 |
| `configs/` | YAML 配置体系，可扩展 |

### 12.2 🔧 重构 (保留但需要改动)

| 路径/组件 | 原因 | 改动方向 |
|-----------|------|---------|
| `src/config.py` | 旧版路径常量，与 `climatenet` 包冲突 | 合并到 `backend/config.py` 或删除 |
| `src/train.py` | 旧版训练脚本，功能被 `benchmark_runner.py` 覆盖 | 删除或简化为 CLI 入口 |
| `src/features.py` | 旧版特征工程，被 `climatenet/features/pipeline.py` 覆盖 | 删除 |
| `src/validation.py` | 旧版划分，被 `split_protocols.py` 覆盖 | 删除 |
| `src/evaluate.py` | 旧版评估，被 `evaluation/metrics.py` 覆盖 | 删除 |
| `src/download_era5.py` | 与 `climatenet/data/era5_download.py` 重复 | 删除旧版 |
| `src/preprocess_era5.py` | 与 `climatenet/data/era5_preprocess.py` 重复 | 删除旧版 |
| `src/physical_features.py` | 与 `climatenet/features/physical.py` 重复 | 删除旧版 |
| `src/explain.py` | SHAP 解释，功能独立 | 迁移到 `climatenet/explain/` 包 |
| `src/plot_results.py` | 可视化 | 迁移到 `climatenet/training/plots.py` (已存在) |
| `src/load_to_db.py` | 数据库导入 | 迁移到 CLI 管理命令 |
| `src/climatenet/models/factory.py` | 与 `model_factory.py` 功能重叠 | 合并为单一工厂 |
| `backend/database.py` | 导入即崩溃 | 改为惰性连接，仅在调用数据库端点时检查 |
| `backend/main.py` | 旧版数据库路由混杂 | 拆分为独立 router 模块或删除 |
| `backend/services/*` | 过薄的 CSV 读取层 | 增加数据库 ORM 支持 (SQLAlchemy 2.0 async) |
| `frontend/` 全部 `.js`/`.vue` | 无 TypeScript | 迁移到 TypeScript (`.ts`/`.vue` with `<script setup lang="ts">`) |
| `dashboard/app.py` (Streamlit) | 与 Vue 前端功能重复 | 评估是否保留，或改为快速原型工具 |

### 12.3 ❌ 删除

| 路径/组件 | 原因 |
|-----------|------|
| `src/make_sample_data.py` | 功能已内联到 `scripts/build_forecasting_dataset.py:_build_synthetic_features()` |
| `src/config.py` (旧版) | 全部被 `climatenet/utils/config.py` 和 `backend/config.py` 取代 |
| `outputs/` 中的已提交实验输出 | 28 MB 的 CSV/PNG 不应该被 git 追踪 |
| `backend/routers/` 中的旧版路由 (comparison.py 的 /ablation-study 与 leaderboard.py 重复) | 合并重复端点 |
| 代码中重复的 `calculate_metrics` (train.py, evaluate.py, train_tcn.py, evaluation/metrics.py) | 统一使用 `climatenet.evaluation.metrics.evaluate_regression` |

### 12.4 ➕ 新增 (升级所需)

| 组件 | 优先级 | 描述 |
|------|--------|------|
| Dockerfile + docker-compose.yml | P0 | 多服务编排: FastAPI, PostgreSQL, Redis, Celery Worker, MinIO, Vue (Nginx) |
| Alembic 迁移 | P0 | 数据库版本管理 |
| Redis + Celery 集成 | P0 | 异步基准测试运行 |
| MinIO 客户端 | P0 | 数据集和模型产物存储 |
| TypeScript 前端迁移 | P1 | `.js` → `.ts`, `.vue` 添加 `lang="ts"` |
| 极端事件指标模块 | P1 | SPI/SPEI, POD, FAR, CSI, ETS, CRPS |
| 新模型: LSTM, Transformer, GNN | P1 | 深度学习架构扩展 |
| 概率预测模型 | P1 | Quantile Regression, Gaussian Process |
| 认证系统 | P2 | JWT / OAuth2 |
| API 速率限制 | P2 | slowapi 或 redis 方案 |
| Prometheus + Grafana 监控 | P2 | 指标导出、仪表盘 |
| 前端组件测试 (Vitest) | P2 | 至少对关键视图的冒烟测试 |
| 后端 API 测试 (httpx) | P2 | 对 34+ 端点的集成测试 |
| Pre-commit 配置 | P2 | ruff + mypy + eslint + prettier |

---

## 附录 A: 文件大小与行数统计

| 目录 | 大小 | Python 行数 | 文件数 |
|------|------|------------|--------|
| `src/` (旧版) | 460K | ~1,300 | 11 |
| `src/climatenet/` | — | ~6,650 | 44 |
| `backend/` | 132K | ~1,900 | 21 |
| `frontend/` | 292K | ~1,500 (JS/Vue) | 20 |
| `tests/` | 132K | ~1,800 | 10+ |
| `scripts/` | 52K | ~600 | 8 |
| `configs/` | 60K | — (YAML) | 10 |
| `docs/` | 56K | — (MD) | 8 |
| `outputs/` | 28M | — | — |
| **总计** | ~29M | ~13,750 | 130+ |

## 附录 B: 技术债务量化

| 债务类型 | 严重程度 | 估计修复工时 |
|---------|---------|------------|
| 双轨代码清理 | 🔴 | 3-5 天 |
| database.py 启动崩溃 | 🔴 | 0.5 天 |
| TypeScript 迁移 | 🟡 | 5-7 天 |
| 实验并行化 | 🟡 | 2-3 天 |
| Docker 化 | 🟡 | 2-3 天 |
| PostgreSQL 真正集成 | 🟡 | 3-5 天 |
| Redis/Celery 集成 | 🟡 | 3-5 天 |
| MinIO 集成 | 🟡 | 1-2 天 |
| CI 优化 | 🟢 | 1 天 |
| 代码质量工具 | 🟢 | 0.5 天 |
| 前端测试 | 🟢 | 3-5 天 |
| 后端 API 测试 | 🟢 | 2-3 天 |
| **总计** | — | **26-45 天** |
