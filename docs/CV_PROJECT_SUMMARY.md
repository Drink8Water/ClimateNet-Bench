# CV Project Summary 草稿

## 一句话定位

构建了一个面向 ERA5-Land 蒸发距平预测的防泄漏时空机器学习 benchmark，
覆盖源数据审计、train-only 特征变换、多随机种子验证和区域分层重复空间验证。

## 中文项目经历 bullet

- 设计 ERA5-Land Sahara 与 East China 2019–2023 蒸发距平预测流程，将
  531.8 万条月度网格记录转换为 478.6 万个六个月滞后样本，并建立可追踪的
  数据、配置、模型和预测 artifact。
- 将月气候态、异常值、事件阈值和标准化参数改为逐 split、仅训练集拟合，
  防止 validation/test 统计量泄漏；为未见区域实现训练集 global-month
  fallback 并记录使用情况。
- 通过时间序列一致性审计定位 ECMWF 已知 accumulated-variable 源产品问题，
  下载并验证 00:00 monthly-by-hour patch，按区域、年月和网格替换
  `ssrd/tp/e`，随后重建并审计 corrected 数据链路。
- 完成 3-seed、18-task corrected robustness benchmark；LightGBM temporal
  RMSE 为 6.309 ± 0.015，相对 random 的退化稳定在约 8.9%–10.1%，并发现
  East China 在 18/18 个区域比较中误差高于 Sahara。
- 发现单次 spatial holdout 受区域构成影响，设计 5 个无 grid leakage 的
  region-stratified spatial folds；LightGBM RMSE 为 7.066 ± 1.093，
  相比 Linear 的 9.782 ± 0.773 降低 27.8%，且在 5/5 folds 上胜出。

## English CV bullet draft

- Built a leakage-aware ERA5-Land benchmark for next-month evaporation
  anomaly forecasting across Sahara and East China, producing 4.79M
  six-month lag samples from 5.32M audited grid-month records.
- Implemented split-specific train-only climatology, anomaly generation,
  event thresholds, feature scaling, and unseen-region fallbacks with
  reproducible metadata and data/config hashes.
- Diagnosed a documented ECMWF accumulated-variable source issue, validated
  and merged corrected hourly-monthly patches, then regenerated and re-audited
  the full dataset before rerunning all reported experiments.
- Evaluated Linear and LightGBM baselines with multi-seed and five-fold
  region-stratified spatial protocols; LightGBM reduced repeated-spatial RMSE
  by 27.8% (7.066 ± 1.093 vs. 9.782 ± 0.773) and won all five folds.

## 可量化结果池

可选用、但不应全部堆在同一版简历中：

- 5,318,160 monthly grid records；4,786,344 six-month lag samples。
- Corrected multi-seed：18/18 tasks completed，0 failed。
- Repeated spatial：10/10 tasks completed，0 failed，5/5 folds 无 grid
  leakage。
- LightGBM random RMSE：5.766 ± 0.020。
- LightGBM temporal RMSE：6.309 ± 0.015；relative degradation
  约 +9.4%。
- LightGBM repeated-spatial RMSE：7.066 ± 1.093。
- Linear repeated-spatial RMSE：9.782 ± 0.773。
- LightGBM repeated-spatial mean RMSE improvement：27.8%，5/5 folds
  优于 Linear。
- East China RMSE 高于 Sahara：18/18 corrected multi-seed regional
  comparisons。
- 完整测试：452 passed，1 skipped（最终整理前的基线）。

## 面试故事线

### 1. 为什么需要 train-only

月气候态、距平、标准化和事件阈值都是由数据估计的参数。如果在完整表上拟合，
validation/test 会改变训练特征和目标定义，导致指标偏乐观。解决方案是先生成
split，再对每个 split 的 train 独立 fit，validation/test 仅 transform，并把
拟合范围、fallback 和统计参数写入 metadata。

### 2. 怎么发现 source-data issue

temporal holdout 出现与 random/spatial 不一致的异常退化。没有直接归因于模型，
而是逐层检查 split、单位换算、bbox、网格、feature engineering 和变量时间序列。
最终发现 2022-09 后 accumulated variables 同时阶跃，并与 ECMWF known issue
周期吻合，说明问题来自上游产品。

### 3. 为什么 corrected 后必须重跑

错误变量参与输入特征和蒸发目标，旧模型指标不能通过后处理修正。保留旧 run
作为审计案例并写入 `source_data_invalid`，随后从 corrected NetCDF 开始重建
processed/physical CSV，以新 hashes 完整重跑 v1 和 multi-seed。

### 4. 为什么 repeated spatial folds 更可信

单次 spatial split 的 seed 2026 test set 有 96.1% Sahara，target variance
更低，因此显得比 random 更容易。通过每个区域内分配空间块，确保每折
train/validation/test 都包含两个区域、同一 grid cell 不跨 partition，并报告
五折 mean/std，可降低一次 block assignment 对结论的支配。

### 5. 传统 ML baseline 的结果

Linear 提供可解释的线性下界，LightGBM 捕获非线性和特征交互。Corrected
multi-seed 与 repeated spatial 都显示 LightGBM 优于 Linear；但结论限定在
Sahara/East China、2019–2023 和当前特征/模型范围内，不包装成全球性结论。

## 使用边界

- 可以说“在两个研究区域和当前协议下”，不要说“全球 ERA5-Land 上”。
- 不引用任何 `source_data_invalid` run 的性能指标。
- temporal 结论应写成 LightGBM 的温和正退化；Linear temporal 相对 random
  并未退化，因此不要概括为“所有模型 temporal 都更难”。
- spatial 正式数字使用 repeated-fold mean/std；single-spatial 仅用于解释
  composition sensitivity。
