# ClimateNet-Bench 4 周 MVP 实施计划

> **目标**: 将 ClimateNet-Bench 升级为陆面水文气候胁迫评估全栈基准平台  
> **定位**: MSc CS/AI/DS 申请作品集核心项目  
> **科学聚焦**: 土壤水分干旱 · 蒸发亏缺 · 复合干热事件  
> **原则**: 每个功能可演示、可复现、可截图放进申请材料  

---

## 总体策略

### 科学范围界定

**本 MVP 聚焦 — 陆面水文气候胁迫 (Land-Surface Hydroclimate Stress)**:

| 评估目标 | 定义 | 标签逻辑 |
|---------|------|---------|
| **土壤水分干旱** | 根区土壤水分异常低于气候态阈值 | `soil_moisture_anomaly < train_P10[calendar_month]` |
| **蒸发亏缺** | 实际蒸发异常显著偏低 | `evaporation_anomaly < train_P10[calendar_month]` |
| **复合干热事件** | 高温与干旱同时发生 | `temperature_anomaly > train_P90[calendar_month]` **且** `soil_moisture_anomaly < train_P10[calendar_month]` |

所有阈值严格从训练集按日历月分层计算——这是反泄漏的核心保证。

**移入 "未来工作" 的内容** (不在本 MVP 实现):
- Rx5day、洪水事件、完整热浪 HWF/HWD 指数
- 完整 ETCCDI 工具箱
- 概率预测 (CRPS、Brier Score)、可靠性图
- GPyTorch、Deep Ensembles
- Prometheus / Grafana、JWT 认证、速率限制
- Kubernetes、PostGIS

### 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 深度学习模型 | **ConvLSTM** (第 3 周可选) | 卷积捕获空间邻域 + LSTM 捕获时间依赖，天然适合网格化气候数据 |
| 传统 ML | **LightGBM** (第 1 周基线) | 梯度提升 SOTA，训练快，早停支持 |
| 基线 | Climatology (train-only), Persistence | 科学基线，不可跳过 |
| 后端 | FastAPI + PostgreSQL + Celery + Redis + MinIO | 全栈能力展示 |
| 前端 | Vue 3 (JavaScript + JSDoc) | 保留现有基础，不迁移 TypeScript |

### ConvLSTM 选型理由

网格化气候数据具有两个关键结构，ConvLSTM 同时建模两者：
- **空间维度**: 邻近网格单元通过大气/陆面过程强相关 — CNN 的平移不变卷积核天然捕获邻域依赖性
- **时间维度**: 土壤水分和蒸发具有记忆效应 (滞后 6 个月) — LSTM 门控机制保留长期依赖

相比普通 LSTM (逐网格单元独立处理，丢失空间上下文)，ConvLSTM 将全连接替换为卷积运算，
输入张量 `(batch, seq_len, channels, H, W)` 直接对应 ERA5-Land 的 0.1°×0.1° 网格。

> **注意**: ConvLSTM 安排在第 3 周作为可选深度学习基线。若第 1-2 周进度延后，可降级为 Future Work，不影响基准平台核心闭环。

---

## 阶段 1: 基准核心 (第 1 周, 5 天)

> **不含深度学习模型。** ConvLSTM 推迟到第 3 周。  
> 本周产出: 训练集阈值、事件标签、检测指标、固定划分清单、LightGBM、mini leaderboard。

### 1.1 代码清理 (安全渐进流程)

**规则**: 不直接删除。每项走: 依赖检查 → 兼容包装 → 弃用注释 → 测试通过后删除。

- `src/config.py` → 逐一迁引用到 `backend/config.py` 或 `climatenet/utils/config.py`，顶部加 `# DEPRECATED`
- `src/train.py`, `src/features.py`, `src/validation.py`, `src/evaluate.py` → 确认新版覆盖后弃用
- `src/download_era5.py`, `src/preprocess_era5.py`, `src/physical_features.py` → 确认 `climatenet/data/` 覆盖后弃用
- `src/make_sample_data.py` → 确认 `scripts/build_forecasting_dataset.py:_build_synthetic_features()` 覆盖后弃用
- `src/explain.py` → 核心逻辑迁入 `src/climatenet/explain/`，旧文件改为 import wrapper
- `src/plot_results.py` → 合并到 `src/climatenet/training/plots.py`
- `src/load_to_db.py` → `mv` 到 `scripts/load_to_db.py`
- **合并两个模型工厂**: `factory.py:build_models()` → `model_factory.py:create_model()`，旧文件弃用
- 统一 `calculate_metrics` → `climatenet.evaluation.metrics.evaluate_regression`
- **修复 `backend/database.py`**: 惰性 `get_engine()`，无 DATABASE_URL 时 `get_db()` 返回 None + log warning，不崩溃
- `outputs/benchmark/`, `outputs/experiments/` 加入 `.gitignore`

**涉及文件**: 弃用 9 个, 修改 10 个, 新建 0 个

### 1.2 水文气候胁迫事件标签构建 (1.5 天)

**标签定义 — 严格 train-only 百分位阈值 (按月分层)**:

| 事件 | 定义 | 阈值来源 |
|------|------|---------|
| `soil_moisture_drought` | `soil_moisture_anomaly < train_P10[calendar_month]` | 训练集每月第 10 百分位 |
| `evaporation_deficit` | `evaporation_anomaly < train_P10[calendar_month]` | 训练集每月第 10 百分位 |
| `compound_hot_dry` | `temperature_anomaly > train_P90[calendar_month]` **且** `soil_moisture_anomaly < train_P10[calendar_month]` | 训练集每月第 90/10 百分位 |

新建 `src/climatenet/evaluation/hydroclimate_labels.py`:
- `fit_event_thresholds(train_df)` → `dict[calendar_month, dict[var, percentile_value]]`
- `build_soil_moisture_drought_label(df, thresholds)` → boolean array
- `build_evaporation_deficit_label(df, thresholds)` → boolean array
- `build_compound_hot_dry_label(df, thresholds)` → boolean array
- `build_all_event_labels(train_df, test_df)` → `dict[event_name, np.ndarray]`

新建 `src/climatenet/evaluation/detection.py`:
- `compute_pod(y_true_label, y_pred_label)` → 零分母返回 `np.nan` + `{"warning": "no observed events"}`
- `compute_far(y_true_label, y_pred_label)` → 零分母返回 `np.nan` + `{"warning": "no predicted events"}`
- `compute_csi(y_true_label, y_pred_label)` → 零分母返回 `np.nan` + `{"warning": "no events in either observed or predicted"}`
- `compute_intensity_bias(y_true, y_pred, y_true_label)` → 仅在事件网格点计算 `mean(y_pred) / mean(y_true)`
- `compute_event_detection_table(results_df, event_types, thresholds)` → DataFrame

**涉及文件**: 新建 3 个, 修改 1 个

### 1.3 训练集气候态 + 固定划分清单 (0.5 天)

- `climatenet/models/climatology.py` — `fit()` 记录 `_fitted_on_train_only=True`；新增 `_validate_train_only()` 严格模式检查
- `climatenet/benchmark/split_protocols.py` — 新增 `export_split_manifest(split_result, output_dir)` → 导出 `train_ids.csv`, `val_ids.csv`, `test_ids.csv`
- 固定清单列: `sample_id, grid_id, region, target_year, target_month`
- `generate_all_splits()` 自动调用 `export_split_manifest()`

**涉及文件**: 修改 2 个

### 1.4 LightGBM 基线增强 (0.5 天)

- 检查 `LightGBMModel` 完整性 (早停、验证集、feature_columns)
- 编写 LightGBM 专用测试: 早停在 patience 轮后生效

**涉及文件**: 修改 1 个

### 1.5 Mini Leaderboard + 基准运行 (1.5 天)

- 新建 `src/climatenet/evaluation/climate_zone_skill.py` → `compute_climate_zone_wise_skill()`
- 更新 `configs/benchmark/smoke_test.yaml` — 模型: `climatology`, `persistence`, `lightgbm` (本周无 ConvLSTM)
- 新建 `scripts/build_mini_leaderboard.py` → 扫描实验 → 输出 3 个 CSV:
  - `leaderboard.csv` (含 `csi_soil_moisture_drought`, `csi_evaporation_deficit`, `csi_compound_hot_dry`)
  - `leaderboard_by_climate_zone.csv`
  - `detection_metrics.csv`

**涉及文件**: 新建 3 个, 修改 2 个

### 验收标准

| # | 标准 | 验证命令 |
|---|------|---------|
| 1 | 弃用文件后测试全绿 | `PYTHONPATH="" python -m pytest tests/ -v` → 全部通过 |
| 2 | 事件标签阈值 train-only 隔离 | `python -m pytest tests/test_hydroclimate_labels.py -v` → 8+ 通过 |
| 3 | POD/FAR/CSI 零分母处理 | `python -m pytest tests/test_detection.py -v` → 8+ 通过 |
| 4 | LightGBM 早停可用 | `python -m pytest tests/test_baselines.py -v -k lightgbm` → 全部通过 |
| 5 | Mini leaderboard 生成 | `python scripts/build_mini_leaderboard.py` → 终端显示排名 + 3 CSV 生成 |
| 6 | 固定划分清单可复现 | 同一 seed 两次运行 → split manifest 的 sample_id 集合一致 |

### 风险
- 事件标签的按月分层阈值在数据量少时可能不稳定 (如某月仅 5 个样本) → 测试用均匀分布的数据
- LightGBM 早停在合成数据上可能不触发 → 测试刻意构造过拟合场景

---

## 阶段 2: 平台后端 (第 2 周, 5 天)

### 目标
实现完整的基准评估平台闭环: prediction.csv 上传 → submission 记录 → Celery 评估 → 指标落库 → 产物存储 → Leaderboard API。

### 数据库设计: 8 张核心表 (含必填字段)

```sql
datasets (
  id SERIAL PK, name TEXT NOT NULL,
  region_config JSONB NOT NULL,        -- {regions: [{name, lat_min, lat_max, lon_min, lon_max, climate_type}]}
  feature_schema JSONB NOT NULL,       -- {feature_columns: [...], target_column: "evaporation_anomaly"}
  time_range JSONB NOT NULL,           -- {start_year, end_year}
  n_samples INT, created_at TIMESTAMPTZ DEFAULT now()
)

benchmark_tasks (
  id SERIAL PK, name TEXT NOT NULL UNIQUE,
  description TEXT,
  target_variable TEXT NOT NULL DEFAULT 'evaporation_anomaly',
  event_types TEXT[] NOT NULL,          -- '{soil_moisture_drought, evaporation_deficit, compound_hot_dry}'
  dataset_id INT FK → datasets.id, created_at TIMESTAMPTZ DEFAULT now()
)

split_protocols (
  id SERIAL PK, benchmark_task_id INT FK → benchmark_tasks.id,
  protocol_name TEXT NOT NULL,          -- 'random'|'spatial_block'|'temporal'|'region_transfer'|'climate_zone_transfer'|'spatiotemporal'
  train_ids_path TEXT, val_ids_path TEXT, test_ids_path TEXT,
  metadata_json JSONB, UNIQUE(benchmark_task_id, protocol_name)
)

models (
  id SERIAL PK, name TEXT NOT NULL UNIQUE,  -- 'climatology_region_monthly'|'persistence'|'lightgbm'|'convlstm'
  model_type TEXT NOT NULL,                 -- 'baseline'|'tree_ensemble'|'deep_learning'
  hyperparams JSONB, created_at TIMESTAMPTZ DEFAULT now()
)

submissions (
  id SERIAL PK, benchmark_task_id INT FK → benchmark_tasks.id,
  model_id INT FK → models.id,
  prediction_file_path TEXT NOT NULL,   -- MinIO key
  status TEXT DEFAULT 'pending',        -- 'pending'|'running'|'done'|'failed'
  created_at TIMESTAMPTZ DEFAULT now()
)

evaluation_runs (
  id SERIAL PK, submission_id INT FK → submissions.id,
  split_protocol_id INT FK → split_protocols.id,
  status TEXT DEFAULT 'pending',        -- 'pending'|'running'|'done'|'failed'
  metrics_json JSONB, error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT now(), completed_at TIMESTAMPTZ
)

metrics (
  id SERIAL PK, evaluation_run_id INT FK → evaluation_runs.id,
  metric_name TEXT NOT NULL,            -- 'rmse'|'mae'|'r2'|'pod'|'far'|'csi'|'intensity_bias'
  metric_value DOUBLE PRECISION,
  event_type TEXT,                      -- NULL for regression; 'soil_moisture_drought' etc.
  climate_zone TEXT,                    -- NULL for global; 'arid'|'monsoon' etc.
  metadata_json JSONB                   -- {"warning": "no observed events"} when NaN
)

artifacts (
  id SERIAL PK, evaluation_run_id INT FK → evaluation_runs.id,
  artifact_type TEXT NOT NULL,          -- 'csv'|'png'|'json'
  storage_path TEXT NOT NULL,           -- MinIO key
  created_at TIMESTAMPTZ DEFAULT now()
)
```

### 任务

#### 2.1 Docker 编排 (1.5 天)
- `Dockerfile` — 多阶段构建 (python:3.11-slim + `libgomp1` for LightGBM)
- `docker-compose.yml` — 6 服务: `api`, `db`, `redis`, `worker`, `minio`, `frontend`
- `docker-compose.override.yml` — 开发模式 (卷挂载、热重载)
- 完善 `.env.example`
- `scripts/entrypoint.sh` — alembic upgrade head → uvicorn

**涉及文件**: 新建 5 个

#### 2.2 PostgreSQL + Alembic (1 天)
- `alembic init alembic` + 配置 `env.py`
- `backend/models.py` — 8 张表 ORM (含上述所有字段、约束、索引)
- 重写 `backend/database.py` — 惰性 `get_engine()` + `get_db()` 依赖注入
- `alembic revision --autogenerate` → `alembic upgrade head`

**涉及文件**: 新建 3 个, 修改 3 个

#### 2.3 基准评估平台闭环 (2 天)

**统一 API 路由前缀 `/api`**:

| 方法 | 路径 | 描述 |
|------|------|------|
| `POST` | `/api/submissions` | 上传 prediction.csv → 创建 submission + N 个 evaluation_runs |
| `GET` | `/api/submissions/{id}` | 返回提交详情 + 关联 evaluation_runs 摘要 |
| `GET` | `/api/evaluation-runs/{id}` | 返回评估运行详情 (指标、状态、产物) |
| `GET` | `/api/evaluation-runs/{id}/status` | 返回 `{status, progress_percent, error_message}` |
| `GET` | `/api/leaderboard` | 排名 (支持 `?metric=&split_protocol=&event_type=&climate_zone=`) |
| `GET` | `/api/artifacts/{id}/download` | 重定向到 MinIO 预签名 URL |

实现链路:
1. `backend/celery_app.py` — Celery 实例，`evaluate_submission_task(submission_id)`
2. `backend/routers/submissions.py` — `POST /api/submissions`
3. `backend/services/evaluation_service.py` — `evaluate_submission()`:
   - 加载 prediction → 合并 ground truth → 对每个 split_protocol 计算
   - 回归指标: MAE/RMSE/R²
   - 事件检测: POD/FAR/CSI/intensity_bias (按 event_type)
   - 气候带分组: climate_zone_wise_skill
   - 写入 metrics 表 → 更新 evaluation_runs.status
4. `backend/services/artifact_service.py` — 产物上传 MinIO + 记录 artifacts 表
5. `backend/routers/evaluation_runs.py` — 状态查询 + 详情
6. `backend/routers/leaderboard.py` — 多维度排名
7. `backend/routers/artifacts.py` — 产物下载

**涉及文件**: 新建 8 个, 修改 3 个

#### 2.4 MinIO 初始化 (0.5 天)
- `backend/minio_client.py` — SDK 封装 + `ensure_buckets(["submissions", "artifacts"])`
- FastAPI startup 自动创建 buckets

**涉及文件**: 新建 1 个, 修改 1 个

### 验收标准

| # | 标准 | 验证命令 |
|---|------|---------|
| 1 | 6 服务健康 | `docker compose up -d && docker compose ps` → 全部 healthy |
| 2 | 8 张表存在 | `docker compose exec db psql -U postgres -d climatenet -c "\dt"` → 列出 8 表 |
| 3 | 迁移成功 | `docker compose exec api alembic current` → 最新 revision |
| 4 | 提交→评估→排行榜 闭环 | `curl -X POST localhost:8000/api/submissions -F "file=@test.csv" -F "benchmark_task_id=1" -F "model_id=1"` → 201；轮询 `GET /api/evaluation-runs/{id}/status` → done；`GET /api/leaderboard` 返回排名 |
| 5 | 产物可见 | `docker compose exec minio mc ls climatenet/artifacts/` → 有文件 |
| 6 | 无 DB 不崩溃 | `DATABASE_URL="" python -c "from backend.main import app"` → 成功 |

### 风险
- 8 张表迁移依赖顺序 (FK 约束) → 先画 ER 图再写 ORM
- Celery worker 需访问同一代码库 → Docker 卷挂载或 pip install -e .

---

## 阶段 3: 前端 + ConvLSTM (可选) (第 3 周, 5 天)

> ConvLSTM 作为可选深度学习基线在本周实现。若第 1-2 周延后，可降级为 Future Work。

### 3.0 ConvLSTM 模型 (可选, 1.5 天)

新建 `src/climatenet/models/convlstm.py`:

```
输入: (batch, seq_len=6, channels=n_features, H, W)
  │
ConvLSTMCell × 2 layers  (3×3 conv, hidden_dim=64, spatial dropout=0.2)
  │
AdaptiveAvgPool2d → (batch, hidden_dim)
  │
Linear(hidden_dim, 64) → ReLU → Dropout(0.2) → Linear(64, 1)
  │
预测蒸发异常
```

- `ConvLSTMCell(nn.Module)` — 单步卷积 LSTM 单元
- `ConvLSTMEncoder(nn.Module)` — 2 层堆叠
- `ConvLSTMModel(ClimateModel)` — fit/predict/get_model_name/save/load
- 早停 (patience=10), `torch.save/load` 序列化
- 在 `model_factory.py` 注册 `"convlstm"`

**涉及文件**: 新建 1 个, 修改 1 个

### 3.1 6 个核心页面 (3 天)

| 页面 | 关键内容 |
|------|---------|
| **Overview** | KPI 卡片 (总实验数、最佳 RMSE、最佳 CSI) + 最近提交列表 |
| **Leaderboard** | 增强筛选 (event_type, split_protocol, climate_zone) + 可排序表格 |
| **ExperimentRunner** | 三步骤: 选择 task+模型 → 上传 CSV → 进度条 → 跳转详情 |
| **EvaluationDetail** | **双栏**: 左侧回归指标 (MAE/RMSE/R² + 散点图 + 残差图)；右侧事件检测 (POD/FAR/CSI/IBias 表格) |
| **ExtremeEventAnalysis** | CSI 柱状图 (多模型) + POD/FAR 散点图 + intensity_bias 柱状图 |
| **ClimateZoneGeneralization** | 模型 × 气候带 RMSE 热力图 + CSI 表格 |

**涉及文件**: 新建 4 个, 修改 3 个

### 3.2 共享组件 (1 天)
- `ProgressBar.vue` — 轮询 `/api/evaluation-runs/{id}/status`，动画过渡
- `EmptyState.vue` — 无数据友好提示
- `SubmissionForm.vue` — 文件拖拽上传 + 下拉选择器
- `MetricsTable.vue` — 通用指标表格，NaN 渲染为 "N/A"
- axios 全局错误拦截器

**涉及文件**: 新建 4 个, 修改 3 个

### 3.3 UI 打磨 (0.5 天)
- API 函数 + 组件 props 添加 JSDoc
- 所有页面三态覆盖 (loading / empty / error)
- 响应式检查 (1366px+)
- 6 张截图

### 验收标准

| # | 标准 | 验证方式 |
|---|------|---------|
| 1 | ConvLSTM 测试通过或全部 skip | `python -m pytest tests/test_convlstm.py -v` → 5+ passed or skipped |
| 2 | 6 页面无 console 错误 | `npm run dev` → 逐一浏览 → console 无 error |
| 3 | 提交→进度→详情流程 | ExperimentRunner 上传 → ProgressBar → 跳转 EvaluationDetail 双栏展示 |
| 4 | 空数据友好 | 无数据时显示 EmptyState，API 错误时显示 ErrorMessage + 重试按钮 |

---

## 阶段 4: 部署、CI、文档与申请包装 (第 4 周, 5 天)

### 4.1 分体 Demo 脚本 (1 天)

| 脚本 | 用途 | 限制 |
|------|------|------|
| `scripts/demo_smoke.py` | CI 用，< 30s | 合成数据 → climatology+persistence+lightgbm (小参数) → 打印 leaderboard。**不训练 ConvLSTM** |
| `scripts/demo_full.py` | 本地演示 | 合成数据 → 全模型 (含 LightGBM 完整参数 + 可选 ConvLSTM) → 全指标 → 写 DB → 生成产物 |
| `scripts/demo.sh` | 一键包装 | `docker compose up -d && sleep 10 && docker compose exec worker python scripts/demo_full.py && curl localhost:8000/api/leaderboard` |

**涉及文件**: 新建 3 个

### 4.2 CI + 覆盖率 (1 天)

- `.github/workflows/ci.yml` — pip 缓存 + ruff + pytest + `demo_smoke.py` (限时 60s)
- `pytest-cov` **不设硬性阈值** (仅报告 %)。核心 benchmark 模块 (`climatenet.evaluation`, `climatenet.benchmark`, `climatenet.models`) 目标 70%
- `.pre-commit-config.yaml` — ruff, trailing-whitespace

**涉及文件**: 新建 1 个, 修改 2 个

### 4.3 代码质量 (1 天)

- `pyproject.toml` → `[tool.ruff]` + `[tool.pytest.ini_options]`
- `ruff check --fix` + `ruff format` 全项目
- `print()` → `logging.getLogger(__name__)` (scripts/ 中的 CLI 输出保留 print)
- `requirements_frozen.txt` (pip freeze + 用途注释)

**涉及文件**: 修改 2 个, 新建 1 个

### 4.4 申请材料 (2 天)

| # | 文件 | 内容要点 |
|---|------|---------|
| 1 | `README.md` (重写) | 定位 + mermaid 架构图 + 截图 + `docker compose up` + CI badge |
| 2 | `docs/SYSTEM_ARCHITECTURE.md` | 数据流图、8 表 ER 图、评估闭环时序图、组件交互图 |
| 3 | `docs/BENCHMARK_REPORT.md` | 任务定义、3 事件定义、6 划分协议、指标公式、基线结果 |
| 4 | `docs/APPLICATION_PORTFOLIO.md` | 动机、技术亮点表、个人贡献、创新点、8 截图 |
| 5 | `docs/DEMO_SCRIPT.md` | 5 分钟旁白稿: 开场→docker up→浏览→提交实验→查看结果→总结 |
| 6 | `docs/CV_BULLETS.md` | 3-4 条英文 CV 要点 |
| 7 | `screenshots/` | 8 张 (6 页面 + docker compose ps + demo.sh 终端) |

**涉及文件**: 新建 6 个, 重写 1 个

### 验收标准

| # | 标准 | 验证命令 |
|---|------|---------|
| 1 | `demo_smoke.py` < 30s | `time python scripts/demo_smoke.py` → < 30s, 退出码 0 |
| 2 | `demo.sh` 完整通过 | `bash scripts/demo.sh` → 退出码 0, curl 返回 leaderboard JSON |
| 3 | CI 全绿 | push GitHub → Actions tab (ruff + pytest + demo_smoke) 全绿 |
| 4 | ruff 零错误 | `ruff check src/ backend/ tests/` → 0 errors |
| 5 | 7 份材料齐全 | `ls README.md docs/SYSTEM_ARCHITECTURE.md docs/BENCHMARK_REPORT.md docs/APPLICATION_PORTFOLIO.md docs/DEMO_SCRIPT.md docs/CV_BULLETS.md screenshots/` |

### 风险
- Docker 镜像拉取慢 (GFW) → pip 清华镜像源
- CI 中 demo_smoke 超时 → 限制合成数据 n_samples=200, lightgbm n_estimators=20

---

## 时间线总览

```
第 1 周                      第 2 周                      第 3 周                      第 4 周
基准核心 (无 DL)             平台后端                     前端 + ConvLSTM (可选)      部署+CI+文档+包装
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ 渐进代码清理  │ Docker 6 服务编排  │ ConvLSTM (可选)       │ demo_smoke + demo_full  │
│ 事件标签+阈值 │ 8 表 DB + Alembic  │ 6 页面重写/新建       │ CI smoketest + ruff     │
│ POD/FAR/CSI   │ /api 闭环 (3.0)    │ 共享组件+MetricsTable │ 代码质量 + frozen reqs  │
│ 固定划分清单  │ Celery evaluate    │ 双栏 EvaluationDetail │ 7 份申请材料            │
│ LightGBM 基线 │ MinIO 存储         │ JSDoc + 截图          │ CV bullets              │
│ mini leaderboard│ Leaderboard API  │                       │                         │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## MSc 申请亮点映射

| 申请维度 | 本项目呈现 |
|---------|-----------|
| **ML/AI** | ConvLSTM 时空深度学习 + LightGBM 梯度提升 + 事件检测指标 (POD/FAR/CSI) |
| **软件工程** | Docker Compose 微服务、PostgreSQL + Alembic 迁移、Celery 异步任务、CI/CD |
| **数据科学** | ERA5-Land 再分析、train-only 事件标签构建、6 种反泄漏划分协议 |
| **全栈开发** | FastAPI `/api` REST + Vue 3 SPA + MinIO 对象存储 + Redis 消息队列 |
| **科研** | 3 类水文气候胁迫事件、climate-zone-wise skill、intensity bias、可复现基准 |
| **影响力** | 气候变化 × ML 交叉领域、可复现评估闭环、完整申请作品集 |

---

## 未来工作 (移出 MVP)

| 类别 | 项目 | 优先级 |
|------|------|--------|
| **科学扩展** | Rx5day 极端降水、完整热浪 HWF/HWD/HWM、洪水事件 | P1 |
| **科学扩展** | 完整 ETCCDI 极端气候指数工具箱 (27 个指数) | P2 |
| **预测** | 概率预测 (CRPS, Brier Score)、可靠性图、分位数回归 | P2 |
| **模型** | GPyTorch 高斯过程、Deep Ensembles、Transformer (PatchTST) | P2 |
| **基础设施** | Prometheus + Grafana 监控、JWT 认证、速率限制 | P1 |
| **基础设施** | Kubernetes Helm Chart、PostGIS 空间查询 | P2 |
| **前端** | TypeScript 完整迁移、Leaflet 空间热图 | P2 |

---

*4 周后你将拥有: 一个聚焦陆面水文气候胁迫评估的全栈基准平台，附带完整的申请作品集材料。*
