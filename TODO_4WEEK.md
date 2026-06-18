# ClimateNet-Bench 4 周 MVP 任务清单

> `[ ]` 待完成  `[~]` 进行中  `[x]` 已完成  
> 🟢 < 1h  🟡 1–3h  🔴 3–8h  

---

## 第 1 周: 基准核心

> **不含深度学习模型。** ConvLSTM 推迟到第 3 周作为可选基线。  
> 本周产出: 训练集阈值、事件标签、检测指标、固定划分清单、LightGBM、mini leaderboard。

### 1.1 代码清理 (安全流程)

**规则**: 不直接删除。每项走 4 步: 依赖检查 → 兼容包装 → 弃用注释 → 测试通过后删除。

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 1.1.1 | **检查** `src/config.py` 的所有导入方 → 列出清单 → 逐一迁到 `backend/config.py` 或 `climatenet/utils/config.py` → 在原文件顶部加 `# DEPRECATED` 注释 → 测试全绿后删除 | 🟡 | [ ] |
| 1.1.2 | **检查** `src/train.py` 调用方 → 确认 `benchmark_runner.py` 覆盖 → 加弃用注释 → 测试全绿后删除 | 🟢 | [ ] |
| 1.1.3 | **检查** `src/features.py` 调用方 → 确认 `climatenet/features/pipeline.py:build_features()` 覆盖 → 加弃用注释 → 删除 | 🟢 | [ ] |
| 1.1.4 | **检查** `src/validation.py` 调用方 → 确认 `split_protocols.py` 覆盖 → 加弃用注释 → 删除 | 🟢 | [ ] |
| 1.1.5 | **检查** `src/evaluate.py` 调用方 → 确认 `evaluation/metrics.py` 覆盖 → 加弃用注释 → 删除 | 🟢 | [ ] |
| 1.1.6 | **检查** `src/download_era5.py` 调用方 → 确认 `climatenet/data/era5_download.py` 覆盖 → 加弃用注释 → 删除 | 🟢 | [ ] |
| 1.1.7 | **检查** `src/preprocess_era5.py` 调用方 → 确认 `climatenet/data/era5_preprocess.py` 覆盖 → 加弃用注释 → 删除 | 🟢 | [ ] |
| 1.1.8 | **检查** `src/physical_features.py` 调用方 → 确认 `climatenet/features/physical.py` 覆盖 → 加弃用注释 → 删除 | 🟢 | [ ] |
| 1.1.9 | **检查** `src/make_sample_data.py` 调用方 → 确认 `scripts/build_forecasting_dataset.py:_build_synthetic_features()` 覆盖 → 加弃用注释 → 删除 | 🟢 | [ ] |
| 1.1.10 | 在 `src/climatenet/explain/` 创建 `__init__.py` → 将 `src/explain.py` 核心逻辑迁入 → 旧文件改为 `from climatenet.explain import *` 包装 → 测试通过后删除旧文件 | 🟡 | [ ] |
| 1.1.11 | `src/plot_results.py` → 功能合并到 `src/climatenet/training/plots.py` → 旧文件加弃用注释 → 删除 | 🟡 | [ ] |
| 1.1.12 | `src/load_to_db.py` → 直接 `mv` 到 `scripts/load_to_db.py` → 更新引用 | 🟢 | [ ] |
| 1.1.13 | **合并两个模型工厂**: 检查 `factory.py` 和 `model_factory.py` 的所有调用方 → 将 `factory.py:build_models()` 的功能合并到 `model_factory.py:create_model()` → `factory.py` 顶部加 `# DEPRECATED — use model_factory.create_model()` → 测试通过后删除 | 🟡 | [ ] |
| 1.1.14 | 扫描 `calculate_metrics` 出现位置 (`train.py`, `evaluate.py`, `train_tcn.py`) → 替换为 `from climatenet.evaluation.metrics import evaluate_regression` → 旧定义加 `# DEPRECATED` | 🟡 | [ ] |
| 1.1.15 | **修复 `backend/database.py`**: 将模块级 `if not DATABASE_URL: raise RuntimeError` 替换为惰性函数 `get_engine()`，仅在首次调用数据库时检查 → 无 DATABASE_URL 时 `get_db()` 返回 `None` 并 log warning | 🟡 | [ ] |
| 1.1.16 | `outputs/benchmark/`, `outputs/experiments/` 加入 `.gitignore` → `git rm --cached` 已追踪文件 | 🟡 | [ ] |
| 1.1.17 | **验收**: `PYTHONPATH="" python -m pytest tests/ -v` → 全部通过 (236+ tests) | 🟡 | [ ] |

### 1.2 水文气候胁迫事件标签构建

**标签定义 (严格 train-only 百分位阈值，按月分层)**:

| 事件 | 定义 | 阈值来源 |
|------|------|---------|
| `soil_moisture_drought` | `soil_moisture_anomaly < train_P10[calendar_month]` | 训练集每月第 10 百分位 |
| `evaporation_deficit` | `evaporation_anomaly < train_P10[calendar_month]` | 训练集每月第 10 百分位 |
| `compound_hot_dry` | `temperature_anomaly > train_P90[calendar_month]` **且** `soil_moisture_anomaly < train_P10[calendar_month]` | 训练集每月第 90/10 百分位 |

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 1.2.1 | 新建 `src/climatenet/evaluation/hydroclimate_labels.py` | 🔴 | [ ] |
| 1.2.2 | 实现 `fit_event_thresholds(train_df)` → 返回 `dict[calendar_month, dict[var, percentile_value]]` | 🔴 | [ ] |
| 1.2.3 | 实现 `build_soil_moisture_drought_label(df, thresholds)` → boolean array，按月匹配 P10 阈值 | 🔴 | [ ] |
| 1.2.4 | 实现 `build_evaporation_deficit_label(df, thresholds)` → boolean array，按月匹配 P10 阈值 | 🟡 | [ ] |
| 1.2.5 | 实现 `build_compound_hot_dry_label(df, thresholds)` → boolean array，温度 > P90 且 土壤水分 < P10 | 🟡 | [ ] |
| 1.2.6 | 实现 `build_all_event_labels(train_df, test_df)` → `dict[event_name, np.ndarray]`，阈值从 train 拟合，标签在 test 上构建 | 🟡 | [ ] |
| 1.2.7 | 编写测试验证: train/test 标签构建隔离 (thresholds 仅从 train 计算，test 从不参与) | 🟡 | [ ] |
| 1.2.8 | 新建 `src/climatenet/evaluation/detection.py` | 🔴 | [ ] |
| 1.2.9 | 实现 `compute_pod(y_true_label, y_pred_label)` — 零分母时返回 `np.nan` 并附带 `{"warning": "no observed events"}` metadata | 🟡 | [ ] |
| 1.2.10 | 实现 `compute_far(y_true_label, y_pred_label)` — 零分母时返回 `np.nan` 并附带 `{"warning": "no predicted events"}` metadata | 🟡 | [ ] |
| 1.2.11 | 实现 `compute_csi(y_true_label, y_pred_label)` — 零分母时返回 `np.nan` 并附带 `{"warning": "no events in either observed or predicted"}` metadata | 🟡 | [ ] |
| 1.2.12 | 实现 `compute_intensity_bias(y_true, y_pred, y_true_label)` — 仅在事件发生的网格点计算 `mean(y_pred) / mean(y_true)` | 🟡 | [ ] |
| 1.2.13 | 实现 `compute_event_detection_table(results_df, event_types, thresholds)` → pandas DataFrame (行=event_type×model, 列=POD/FAR/CSI/IBias) | 🟡 | [ ] |
| 1.2.14 | 更新 `src/climatenet/evaluation/__init__.py` — 导出新模块 | 🟢 | [ ] |
| 1.2.15 | 编写 `tests/test_hydroclimate_labels.py` (8+ 测试: 阈值拟合、按月分层、train/test隔离、3种事件标签、边界值) | 🟡 | [ ] |
| 1.2.16 | 编写 `tests/test_detection.py` (8+ 测试: POD/FAR/CSI 正常值、零分母→NaN、intensity_bias、批量评估表) | 🟡 | [ ] |
| 1.2.17 | **验收**: `python -m pytest tests/test_hydroclimate_labels.py tests/test_detection.py -v` → 16+ 通过 | 🟡 | [ ] |

### 1.3 训练集气候态 + 固定划分清单

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 1.3.1 | `climatenet/models/climatology.py` — 在 `fit()` 中记录 `_fitted_on_train_only=True` 标记；增加 `_validate_train_only()` 检查: 若 detect 任何 test 数据混入则 raise | 🟡 | [ ] |
| 1.3.2 | `climatenet/benchmark/split_protocols.py` — 添加 `export_split_manifest(split_result, output_dir)` → 导出 3 个 CSV: `train_ids.csv`, `val_ids.csv`, `test_ids.csv` | 🟡 | [ ] |
| 1.3.3 | 固定划分清单包含列: `sample_id, grid_id, region, target_year, target_month` | 🟢 | [ ] |
| 1.3.4 | 更新 `generate_all_splits()` 调用 `export_split_manifest()` 自动导出 | 🟢 | [ ] |
| 1.3.5 | 编写测试: 同一 seed 两次运行 → split manifest 的 sample_id 集合完全一致 | 🟡 | [ ] |
| 1.3.6 | 编写测试: train_only 验证在训练集混入 test 数据时抛出异常 | 🟡 | [ ] |

### 1.4 LightGBM 基线模型增强

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 1.4.1 | 检查 `climatenet/models/tree_models.py` 中 `LightGBMModel` 的完整性 | 🟢 | [ ] |
| 1.4.2 | 确保 LightGBM 支持 `val_df` 参数 (早停) | 🟡 | [ ] |
| 1.4.3 | 确保 LightGBM 支持 `feature_columns` 参数 → 传递给 `LGBMRegressor.fit(X[feature_columns], y)` | 🟢 | [ ] |
| 1.4.4 | 编写 LightGBM 专用测试: 早停生效 (train loss 下降, val loss 在 patience 轮后停止) | 🟡 | [ ] |
| 1.4.5 | **验收**: `python -m pytest tests/test_baselines.py -v -k lightgbm` → 全部通过 | 🟡 | [ ] |

### 1.5 Mini Leaderboard + 基准运行

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 1.5.1 | 新建 `src/climatenet/evaluation/climate_zone_skill.py` | 🟡 | [ ] |
| 1.5.2 | 实现 `compute_climate_zone_wise_skill(results_df, metric='rmse')` → `dict[climate_zone, dict[model_name, float]]` | 🟡 | [ ] |
| 1.5.3 | 更新 `configs/benchmark/smoke_test.yaml` — 模型列表: `climatology`, `persistence`, `lightgbm` (本周不含 ConvLSTM) | 🟡 | [ ] |
| 1.5.4 | 新建 `scripts/build_mini_leaderboard.py` — 扫描实验目录 → 计算基础指标 + 事件检测指标 + 气候带 skill → 输出 3 个 CSV | 🔴 | [ ] |
| 1.5.5 | 输出文件: `leaderboard.csv` (含 csi_soil_moisture_drought, csi_evaporation_deficit, csi_compound_hot_dry 列), `leaderboard_by_climate_zone.csv`, `detection_metrics.csv` | 🟡 | [ ] |
| 1.5.6 | 终端输出: 排名表 (top-5 by RMSE) + 事件检测摘要 (best CSI per event type) | 🟡 | [ ] |
| 1.5.7 | 编写 `tests/test_climate_zone_skill.py` (4+ 测试) | 🟡 | [ ] |
| 1.5.8 | **验收**: `python scripts/build_mini_leaderboard.py` → 终端显示排名 + 3 个 CSV 生成 | 🟡 | [ ] |
| 1.5.9 | **验收**: 冒烟测试输出包含 `climatology`, `persistence`, `lightgbm` 三模型行 | 🟡 | [ ] |

---

## 第 2 周: 平台后端

### 2.1 Docker 编排

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 2.1.1 | 编写 `Dockerfile` — 多阶段 (python:3.11-slim + `libgomp1` for LightGBM) | 🟡 | [ ] |
| 2.1.2 | 编写 `docker-compose.yml` — 6 服务: `api`, `db`, `redis`, `worker`, `minio`, `frontend` | 🔴 | [ ] |
| 2.1.3 | 编写 `docker-compose.override.yml` — 开发模式 (卷挂载源码, uvicorn --reload, vite HMR) | 🟡 | [ ] |
| 2.1.4 | 完善 `.env.example` — `DATABASE_URL`, `REDIS_URL`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `MINIO_ENDPOINT` | 🟢 | [ ] |
| 2.1.5 | 编写 `scripts/entrypoint.sh` — `alembic upgrade head && uvicorn backend.main:app --host 0.0.0.0 --port 8000` | 🟡 | [ ] |
| 2.1.6 | **验收**: `docker compose up -d && docker compose ps` → 6 个服务 `status=healthy` | 🟡 | [ ] |

### 2.2 PostgreSQL + Alembic (8 张核心表)

**8 张表及必填字段**:

```sql
datasets (
  id SERIAL PK,
  name TEXT NOT NULL,
  region_config JSONB NOT NULL,        -- {regions: [{name, lat_min, lat_max, lon_min, lon_max, climate_type}]}
  feature_schema JSONB NOT NULL,       -- {feature_columns: [...], target_column: "evaporation_anomaly"}
  time_range JSONB NOT NULL,           -- {start_year, end_year}
  n_samples INT,
  created_at TIMESTAMPTZ DEFAULT now()
)

benchmark_tasks (
  id SERIAL PK,
  name TEXT NOT NULL UNIQUE,           -- e.g. "evap_anomaly_v1"
  description TEXT,
  target_variable TEXT NOT NULL DEFAULT 'evaporation_anomaly',
  event_types TEXT[] NOT NULL,          -- '{soil_moisture_drought, evaporation_deficit, compound_hot_dry}'
  dataset_id INT FK → datasets.id,
  created_at TIMESTAMPTZ DEFAULT now()
)

split_protocols (
  id SERIAL PK,
  benchmark_task_id INT FK → benchmark_tasks.id,
  protocol_name TEXT NOT NULL,          -- 'random' | 'spatial_block' | 'temporal' | 'region_transfer' | 'climate_zone_transfer' | 'spatiotemporal'
  train_ids_path TEXT,                  -- MinIO key or local path
  val_ids_path TEXT,
  test_ids_path TEXT,
  metadata_json JSONB,
  UNIQUE(benchmark_task_id, protocol_name)
)

models (
  id SERIAL PK,
  name TEXT NOT NULL UNIQUE,            -- 'climatology_region_monthly' | 'persistence' | 'lightgbm' | 'convlstm'
  model_type TEXT NOT NULL,             -- 'baseline' | 'tree_ensemble' | 'deep_learning'
  hyperparams JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
)

submissions (
  id SERIAL PK,
  benchmark_task_id INT FK → benchmark_tasks.id,
  model_id INT FK → models.id,
  prediction_file_path TEXT NOT NULL,   -- MinIO key
  status TEXT DEFAULT 'pending',        -- 'pending' | 'running' | 'done' | 'failed'
  created_at TIMESTAMPTZ DEFAULT now()
)

evaluation_runs (
  id SERIAL PK,
  submission_id INT FK → submissions.id,
  split_protocol_id INT FK → split_protocols.id,
  status TEXT DEFAULT 'pending',        -- 'pending' | 'running' | 'done' | 'failed'
  metrics_json JSONB,
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  completed_at TIMESTAMPTZ
)

metrics (
  id SERIAL PK,
  evaluation_run_id INT FK → evaluation_runs.id,
  metric_name TEXT NOT NULL,            -- 'rmse' | 'mae' | 'r2' | 'pod' | 'far' | 'csi' | 'intensity_bias'
  metric_value DOUBLE PRECISION,
  event_type TEXT,                      -- NULL for regression metrics; 'soil_moisture_drought' etc. for detection
  climate_zone TEXT,                    -- NULL for global; 'arid' | 'monsoon' etc. for zonal
  metadata_json JSONB                   -- e.g. {"warning": "no observed events"} when metric_value is NaN
)

artifacts (
  id SERIAL PK,
  evaluation_run_id INT FK → evaluation_runs.id,
  artifact_type TEXT NOT NULL,          -- 'csv' | 'png' | 'json'
  storage_path TEXT NOT NULL,           -- MinIO key
  created_at TIMESTAMPTZ DEFAULT now()
)
```

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 2.2.1 | 安装 `alembic` (psycopg2-binary 已有) | 🟢 | [ ] |
| 2.2.2 | `alembic init alembic` → 配置 `alembic.ini` + `env.py` (导入 Base from backend.models) | 🟡 | [ ] |
| 2.2.3 | 创建 `backend/models.py` — 8 张表 ORM 模型，含上述所有必填字段 + 约束 + 索引 | 🔴 | [ ] |
| 2.2.4 | 重写 `backend/database.py` — `get_engine()` 惰性创建；`get_db()` 依赖注入；无 `DATABASE_URL` 时返回 `None` 并 log warning | 🟡 | [ ] |
| 2.2.5 | `alembic revision --autogenerate -m "init_8_tables"` → 检查迁移脚本 → `alembic upgrade head` | 🟡 | [ ] |
| 2.2.6 | `backend/main.py` — DB 可用时激活 `/api/*` 数据库端点，不可用时端点返回 `{"detail": "database not configured"}` (503) | 🟡 | [ ] |
| 2.2.7 | **验收**: `docker compose exec db psql -U postgres -d climatenet -c "\dt"` → 列出 8 张表 | 🟡 | [ ] |
| 2.2.8 | **验收**: `docker compose exec api alembic current` → 显示最新 revision | 🟡 | [ ] |

### 2.3 基准评估平台闭环 (统一 `/api` 前缀)

**API 路由规范**:

| 方法 | 路径 | 描述 |
|------|------|------|
| `POST` | `/api/submissions` | 上传 prediction.csv，创建 submission + N 个 evaluation_runs |
| `GET` | `/api/submissions/{submission_id}` | 返回提交详情 + 关联 evaluation_runs 摘要 |
| `GET` | `/api/evaluation-runs/{run_id}` | 返回评估运行详情 (指标、状态) |
| `GET` | `/api/evaluation-runs/{run_id}/status` | 返回 `{status, progress_percent, error_message}` |
| `GET` | `/api/leaderboard` | 排名 (支持 `?metric=&split_protocol=&event_type=&climate_zone=`) |
| `GET` | `/api/artifacts/{artifact_id}/download` | 下载评估产物 (重定向到 MinIO 预签名 URL) |

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 2.3.1 | 安装 `celery[redis]` | 🟢 | [ ] |
| 2.3.2 | 创建 `backend/celery_app.py` — Celery 实例 (`broker=REDIS_URL`, `result_backend=REDIS_URL`) | 🟡 | [ ] |
| 2.3.3 | 定义 `evaluate_submission_task(submission_id)` — 核心评估 Celery 任务 | 🔴 | [ ] |
| 2.3.4 | 创建 `backend/routers/submissions.py` — `POST /api/submissions` (multipart: prediction.csv + benchmark_task_id + model_id) | 🔴 | [ ] |
| 2.3.5 | 创建 `backend/services/evaluation_service.py` — 评估逻辑 | 🔴 | [ ] |
| 2.3.6 | `evaluate_submission()` 内部步骤: (a) 加载 prediction.csv (b) 合并 ground truth (c) 对每个 split_protocol 计算指标 (d) 回归指标: MAE/RMSE/R² (e) 事件检测指标: POD/FAR/CSI/intensity_bias (f) 按 climate_zone 分组计算 (g) 写入 metrics 表 (h) 更新 evaluation_runs.status='done' | 🔴 | [ ] |
| 2.3.7 | 创建 `backend/services/artifact_service.py` — 生成 CSV/JSON 产物 → 上传 MinIO → 写入 artifacts 表 | 🟡 | [ ] |
| 2.3.8 | 创建 `backend/routers/evaluation_runs.py` — `GET /api/evaluation-runs/{run_id}`, `GET /api/evaluation-runs/{run_id}/status` | 🟡 | [ ] |
| 2.3.9 | 重写 `backend/routers/leaderboard.py` — `GET /api/leaderboard` (支持 query params: `metric`, `split_protocol`, `event_type`, `climate_zone`) | 🟡 | [ ] |
| 2.3.10 | 创建 `backend/routers/artifacts.py` — `GET /api/artifacts/{artifact_id}/download` | 🟡 | [ ] |
| 2.3.11 | 在 `backend/main.py` 注册所有新 router (prefix 统一为 `/api`) | 🟢 | [ ] |
| 2.3.12 | 编写 `tests/test_api_submissions.py` (3+ 测试: 上传成功、缺少字段 422、无效 benchmark_task_id 404) | 🟡 | [ ] |
| 2.3.13 | **验收**: `curl -X POST localhost:8000/api/submissions -F "file=@test_prediction.csv" -F "benchmark_task_id=1" -F "model_id=1"` → 201, 返回 `{submission_id, evaluation_run_ids}` | 🟡 | [ ] |
| 2.3.14 | **验收**: 轮询 `GET /api/evaluation-runs/{run_id}/status` → `pending → running → done` | 🟡 | [ ] |
| 2.3.15 | **验收**: `curl "localhost:8000/api/leaderboard?metric=csi&event_type=soil_moisture_drought"` → 返回排名 JSON | 🟡 | [ ] |

### 2.4 MinIO 初始化

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 2.4.1 | 安装 `minio` Python SDK | 🟢 | [ ] |
| 2.4.2 | 创建 `backend/minio_client.py` — 封装 MinIO + `ensure_buckets(["submissions", "artifacts"])` | 🟡 | [ ] |
| 2.4.3 | FastAPI startup event 中调用 `ensure_buckets()` | 🟢 | [ ] |
| 2.4.4 | **验收**: `docker compose exec minio mc ls climatenet/submissions/ && docker compose exec minio mc ls climatenet/artifacts/` → 两个 bucket 存在 | 🟡 | [ ] |

---

## 第 3 周: 前端 + ConvLSTM (可选深度学习基线)

> ConvLSTM 作为可选模型在本周实现。若第 1-2 周进度延后，ConvLSTM 可降级为 Future Work，不影响基准平台闭环。

### 3.0 ConvLSTM 模型 (可选, 1.5 天)

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 3.0.1 | 新建 `src/climatenet/models/convlstm.py` | 🔴 | [ ] |
| 3.0.2 | 实现 `ConvLSTMCell(nn.Module)` — 单步卷积 LSTM (3×3 conv, hidden_dim=64, 输入 `(batch, channels, H, W)`) | 🔴 | [ ] |
| 3.0.3 | 实现 `ConvLSTMEncoder(nn.Module)` — 2 层堆叠 + spatial dropout=0.2 | 🔴 | [ ] |
| 3.0.4 | 实现 `ConvLSTMModel(ClimateModel)` — `fit()` 将 3D 序列 `(batch, seq_len, n_features)` reshape 为 `(batch, seq_len, n_features, H, W)` 伪网格；`predict()` 返回 1-D array；`get_model_name()` 返回 `"convlstm"` | 🔴 | [ ] |
| 3.0.5 | 早停 (patience=10, monitor val_loss, `torch.save` best model) | 🟡 | [ ] |
| 3.0.6 | `save(path)` → `torch.save(model.state_dict(), path)`；`load(path)` → `model.load_state_dict(torch.load(path))` | 🟡 | [ ] |
| 3.0.7 | 在 `model_factory.py` 注册 `"convlstm"` → `create_model("convlstm", kwargs)` | 🟢 | [ ] |
| 3.0.8 | 编写 `tests/test_convlstm.py` (5+ 测试: 输出形状、fit 不崩溃、save/load 往返、早停在 patience 后触发、get_model_name) | 🟡 | [ ] |
| 3.0.9 | **验收**: `python -m pytest tests/test_convlstm.py -v` → 5+ 通过 (若未实现则全部 skip) | 🟡 | [ ] |

### 3.1 6 个核心页面

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 3.1.1 | 重写 `frontend/src/views/Overview.vue` — KPI 卡片 (总实验数、最佳 RMSE、最佳 CSI) + 最近提交列表 (轮询 `/api/submissions`) | 🟡 | [ ] |
| 3.1.2 | 重写 `frontend/src/views/Leaderboard.vue` — 增强筛选 (按 `event_type`, `split_protocol`, `climate_zone`) + 可排序表格 | 🔴 | [ ] |
| 3.1.3 | 新建 `frontend/src/views/ExperimentRunner.vue` — 三步骤: (1) 选择 benchmark_task + model (2) 拖拽/选择 prediction.csv (3) 提交 → 进度条轮询 `/api/evaluation-runs/{id}/status` → 完成后跳转详情 | 🔴 | [ ] |
| 3.1.4 | 新建 `frontend/src/views/EvaluationDetail.vue` — **同时展示回归指标和事件检测指标**: 左侧: 回归卡片 (MAE/RMSE/R²) + 预测 vs 实际散点图 + 残差直方图；右侧: 事件检测表格 (POD/FAR/CSI/IBias × 3 种事件类型) | 🔴 | [ ] |
| 3.1.5 | 新建 `frontend/src/views/ExtremeEventAnalysis.vue` — 按事件类型切换: CSI 对比柱状图 (多模型) + POD/FAR 散点图 + intensity_bias 柱状图 | 🔴 | [ ] |
| 3.1.6 | 新建 `frontend/src/views/ClimateZoneGeneralization.vue` — 热力图 (模型 × 气候带 RMSE) + 按气候带 CSI 表格 | 🟡 | [ ] |
| 3.1.7 | 更新 `frontend/src/router/index.js` — 6 条路由 (`/`, `/leaderboard`, `/run`, `/evaluation/:id`, `/extremes`, `/climate-zones`) | 🟢 | [ ] |
| 3.1.8 | 更新 `frontend/src/api/climateApi.js` — 添加 `submitPrediction()`, `fetchEvaluationRun()`, `fetchEvaluationRunStatus()`, `fetchLeaderboard(params)`, `fetchArtifactDownload()` | 🟡 | [ ] |
| 3.1.9 | `fetchEvaluationRunStatus()` 实现轮询逻辑: 每 2 秒请求直到 status=done/failed，最大 150 次 (5 分钟超时) | 🟡 | [ ] |
| 3.1.10 | **验收**: `npm run dev` → 逐一浏览 6 个页面 → console 无 error + 无 unresolved promise | 🟡 | [ ] |

### 3.2 共享组件

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 3.2.1 | 新建 `frontend/src/components/common/ProgressBar.vue` — props: `status`, `progressPercent`；动画过渡；done/failed 颜色切换 | 🟡 | [ ] |
| 3.2.2 | 新建 `frontend/src/components/common/EmptyState.vue` — props: `title`, `description`, `icon` | 🟢 | [ ] |
| 3.2.3 | 新建 `frontend/src/components/common/SubmissionForm.vue` — 文件拖拽上传区 + benchmark_task/model 下拉选择器 + 提交按钮 (loading 态) | 🟡 | [ ] |
| 3.2.4 | 新建 `frontend/src/components/common/MetricsTable.vue` — 通用指标表格组件，props: `columns`, `rows`, `sortable`，支持 NaN 渲染为 "N/A" | 🟡 | [ ] |
| 3.2.5 | 添加 axios 全局响应拦截器 — 非 2xx 显示 toast 错误通知 | 🟡 | [ ] |
| 3.2.6 | 更新侧边栏 `Sidebar.vue` — 6 个导航项 (含图标) | 🟢 | [ ] |
| 3.2.7 | **验收**: ExperimentRunner 选择 task + 模型 → 上传 CSV → 提交 → ProgressBar 进度动画 → 完成后自动跳转 `/evaluation/:id` | 🟡 | [ ] |

### 3.3 UI 打磨

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 3.3.1 | 为 `climateApi.js` 所有导出函数添加 JSDoc (`@param`, `@returns`, `@throws`) | 🟡 | [ ] |
| 3.3.2 | 为共享组件 props 添加 JSDoc + `required` 标记 | 🟡 | [ ] |
| 3.3.3 | 所有页面覆盖三态: loading (skeleton/spinner) / empty (EmptyState) / error (ErrorMessage + retry button) | 🟡 | [ ] |
| 3.3.4 | 响应式检查 (1366px+ 主目标，确保侧边栏 + 主内容区不错位) | 🟡 | [ ] |
| 3.3.5 | **验收**: 无数据时每个页面显示 EmptyState 而非空白；API 错误时显示 ErrorMessage + 重试按钮 | 🟡 | [ ] |

### 3.4 截图

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 3.4.1 | 截图: Overview (KPI + 提交列表) | 🟢 | [ ] |
| 3.4.2 | 截图: Leaderboard (按 soil_moisture_drought 的 CSI 筛选) | 🟢 | [ ] |
| 3.4.3 | 截图: ExperimentRunner (进度条 60% 状态) | 🟢 | [ ] |
| 3.4.4 | 截图: EvaluationDetail (回归指标 + 事件检测指标 双栏) | 🟢 | [ ] |
| 3.4.5 | 截图: ExtremeEventAnalysis (CSI 柱状图) | 🟢 | [ ] |
| 3.4.6 | 截图: ClimateZoneGeneralization (RMSE 热力图) | 🟢 | [ ] |

---

## 第 4 周: 部署、CI、文档与申请包装

### 4.1 分体 Demo 脚本

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 4.1.1 | 编写 `scripts/demo_smoke.py` — CI 用，< 30 秒，**不训练深度学习模型**: 构建合成数据 → 训练 climatology + persistence + lightgbm (小参数量) → 计算指标 → 打印 leaderboard | 🔴 | [ ] |
| 4.1.2 | 编写 `scripts/demo_full.py` — 本地演示用: 合成数据 → 训练全部模型 (含 LightGBM 完整参数 + 可选 ConvLSTM) → 全指标 → 写 DB → 生成所有产物 → 打印 leaderboard | 🔴 | [ ] |
| 4.1.3 | 编写 `scripts/demo.sh` — 包装脚本: `docker compose up -d && sleep 10 && docker compose exec worker python scripts/demo_full.py && curl -s localhost:8000/api/leaderboard` | 🟡 | [ ] |
| 4.1.4 | **验收**: `time python scripts/demo_smoke.py` → < 30s, 退出码 0 | 🟡 | [ ] |
| 4.1.5 | **验收**: `bash scripts/demo.sh` → 退出码 0, curl 返回 leaderboard JSON | 🟡 | [ ] |

### 4.2 CI Smoketest + 覆盖率

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 4.2.1 | 更新 `.github/workflows/ci.yml` — 添加 `cache: 'pip'` | 🟢 | [ ] |
| 4.2.2 | CI 添加 ruff 检查步骤 (`ruff check src/ backend/ tests/`) | 🟡 | [ ] |
| 4.2.3 | CI 添加 pytest 步骤: `pytest tests/ -v --cov=climatenet --cov-report=term` — **不设硬性阈值** (仅报告覆盖率%) | 🟡 | [ ] |
| 4.2.4 | CI 添加 demo_smoke 步骤: `python scripts/demo_smoke.py` (限时 60s) | 🟡 | [ ] |
| 4.2.5 | 创建 `.pre-commit-config.yaml` (ruff, trailing-whitespace, end-of-file-fixer) | 🟢 | [ ] |
| 4.2.6 | 在 `pyproject.toml` 添加覆盖率配置: 核心 benchmark 模块 (`climatenet.evaluation`, `climatenet.benchmark`, `climatenet.models`) 目标 70%；整体不设硬阈值 | 🟡 | [ ] |
| 4.2.7 | **验收**: push 到 GitHub → Actions tab 全绿 (ruff + pytest + demo_smoke) | 🟡 | [ ] |

### 4.3 代码质量

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 4.3.1 | `pyproject.toml` 添加 `[tool.ruff]` (line-length=100, select=["E","F","I","N","W","B","C4"]) | 🟢 | [ ] |
| 4.3.2 | `pyproject.toml` 添加 `[tool.pytest.ini_options]` (testpaths=["tests"], addopts="-v --tb=short") | 🟢 | [ ] |
| 4.3.3 | `ruff check --fix` + `ruff format` 全项目 | 🟡 | [ ] |
| 4.3.4 | 扫描 `print()` → 替换为 `logging.getLogger(__name__).info/warning/error` (scripts/ 中的 CLI 输出保留 print) | 🟡 | [ ] |
| 4.3.5 | 生成 `requirements_frozen.txt` — `pip freeze` + 每行注释用途 (格式: `pandas==3.0.3  # 数据处理核心`) | 🟢 | [ ] |
| 4.3.6 | **验收**: `ruff check src/ backend/ tests/` → 0 errors | 🟡 | [ ] |

### 4.4 申请材料 (7 份)

| ID | 任务 | 难度 | 状态 |
|----|------|------|------|
| 4.4.1 | 重写 `README.md` — 项目定位 + ASCII 架构图 (mermaid) + 6 张截图 + `docker compose up` 快速开始 + 复现步骤 + CI badge | 🔴 | [ ] |
| 4.4.2 | 新建 `docs/SYSTEM_ARCHITECTURE.md` — 数据流图 (ERA5 → features → forecasting samples → models → predictions → evaluation → leaderboard)；8 表 ER 图 (ASCII or mermaid)；评估闭环时序图 (submission → Celery → metrics → artifacts)；组件交互图 (Vue ↔ FastAPI ↔ PostgreSQL ↔ Redis ↔ Celery ↔ MinIO) | 🔴 | [ ] |
| 4.4.3 | 新建 `docs/BENCHMARK_REPORT.md` — 任务定义；3 种事件定义 (含公式)；6 种划分协议说明；4 模型阵容 (Climatology/Persistence/LightGBM/ConvLSTM)；指标公式 (MAE/RMSE/R²/POD/FAR/CSI/IBias)；基线结果表 (合成数据)；气候带泛化分析 | 🔴 | [ ] |
| 4.4.4 | 新建 `docs/APPLICATION_PORTFOLIO.md` — 项目动机 (气候变化 × ML 交叉)；技术栈亮点表；个人贡献清单 (bullet points)；难度与创新点；8 张截图 (每张附说明) | 🔴 | [ ] |
| 4.4.5 | 新建 `docs/DEMO_SCRIPT.md` — 5 分钟演示旁白稿: 0:00 开场 (背景+动机) → 1:00 `docker compose up` → 2:00 前端浏览 (6 页面快速过) → 3:00 提交实验 (上传 CSV → 进度条 → 结果) → 4:00 极端事件 + 气候带分析 → 4:30 总结 | 🟡 | [ ] |
| 4.4.6 | 新建 `docs/CV_BULLETS.md` — 3-4 条英文 CV 要点，每条 1-2 行 (e.g. "Built a full-stack benchmark platform for land-surface hydroclimate stress evaluation using FastAPI, Vue 3, PostgreSQL, Celery, MinIO, and PyTorch ConvLSTM on ERA5-Land reanalysis data.") | 🟡 | [ ] |
| 4.4.7 | 收集 `screenshots/` 8 张 (6 页面 + docker-compose ps 终端 + demo.sh 终端输出) | 🟡 | [ ] |
| 4.4.8 | **验收**: `ls README.md docs/SYSTEM_ARCHITECTURE.md docs/BENCHMARK_REPORT.md docs/APPLICATION_PORTFOLIO.md docs/DEMO_SCRIPT.md docs/CV_BULLETS.md screenshots/` → 7 项齐全 | 🟡 | [ ] |

---

## 统计摘要

| 阶段 | 新建文件 | 修改文件 | 弃用删除文件 | 任务数 | 估计天数 |
|------|---------|---------|------------|--------|---------|
| 第 1 周 | 5 | 10 | 9 (渐进) | 43 | 5 |
| 第 2 周 | 14 | 7 | 0 | 29 | 5 |
| 第 3 周 | 8 | 5 | 0 | 32 | 5 |
| 第 4 周 | 7 | 5 | 0 | 25 | 5 |
| **总计** | **34** | **27** | **9** | **129** | **20** |

---

## 验收检查清单 (最终)

| # | 验证命令 | 期望结果 |
|---|---------|---------|
| V1 | `PYTHONPATH="" python -m pytest tests/ -v` | 全部通过 |
| V2 | `ruff check src/ backend/ tests/` | 0 errors |
| V3 | `time python scripts/demo_smoke.py` | < 30s, 退出码 0 |
| V4 | `docker compose up -d && docker compose ps` | 6 个服务 healthy |
| V5 | `bash scripts/demo.sh` | 退出码 0, leaderboard JSON 有数据 |
| V6 | `curl -X POST localhost:8000/api/submissions -F "file=@test_prediction.csv" -F "benchmark_task_id=1" -F "model_id=1"` | 201, 返回 submission_id |
| V7 | `curl "localhost:8000/api/leaderboard?metric=csi&event_type=soil_moisture_drought"` | 200, 返回排名 JSON |
| V8 | `ls docs/SYSTEM_ARCHITECTURE.md docs/BENCHMARK_REPORT.md docs/APPLICATION_PORTFOLIO.md docs/DEMO_SCRIPT.md docs/CV_BULLETS.md screenshots/` | 6 项齐全 |

---

*每完成一项请勾选 `[x]`。最终验收时逐一执行 V1–V8。*
