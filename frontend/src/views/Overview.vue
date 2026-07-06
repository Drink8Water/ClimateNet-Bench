<script setup>
import { ref, onMounted } from 'vue'
import { fetchBenchmarkSummary, fetchBenchmarkTask, fetchBenchmarkRegions, fetchBenchmarkSplits } from '../api/climateApi'
import MetricCard from '../components/common/MetricCard.vue'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'
import { useI18n } from '../i18n'

const summary = ref(null); const task = ref(null)
const regions = ref([]); const splits = ref([])
const loading = ref(true); const error = ref('')
const { pick } = useI18n()

onMounted(async () => {
  try {
    const [s, t, r, sp] = await Promise.all([
      fetchBenchmarkSummary(), fetchBenchmarkTask(),
      fetchBenchmarkRegions(), fetchBenchmarkSplits()
    ])
    summary.value = s; task.value = t; regions.value = r; splits.value = sp
  } catch (e) { error.value = e.message } finally { loading.value = false }
})
</script>

<template>
  <div v-if="loading"><LoadingState :message="pick('正在加载基准概览...', 'Loading benchmark overview...')" /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-6">
    <section class="page-hero p-6 lg:p-8">
      <div class="grid grid-cols-1 gap-6 lg:grid-cols-[1.15fr_0.85fr] lg:items-end">
        <div>
          <p class="text-xs font-semibold text-[var(--color-muted)]">EvapAnomaly-Forecast-v1</p>
          <h1 class="mt-2 max-w-3xl text-3xl font-bold leading-tight text-[var(--color-text)]">
            {{ pick('面向气候模型泛化能力的实验评测工作台', 'A benchmark workbench for climate model generalization') }}
          </h1>
          <p class="mt-3 max-w-2xl text-sm leading-6 text-[var(--color-muted)]">
            {{ pick('围绕下月陆面蒸散异常预测，比较模型在未见网格、未来年份和跨气候区迁移时的稳健性。', 'Compare next-month land evaporation anomaly forecasts across unseen grids, future years, and climate-region transfer settings.') }}
          </p>
        </div>
        <div class="analysis-panel p-4">
          <p class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('核心问题', 'Central question') }}</p>
          <p class="mt-2 text-base font-semibold leading-6">
            {{ pick('模型是真的学到了可迁移气候关系，还是只记住了特定区域和年份的统计模式？', 'Do models learn transferable climate relationships, or memorize regional and temporal patterns?') }}
          </p>
        </div>
      </div>
    </section>

    <div class="grid grid-cols-2 lg:grid-cols-5 gap-4">
      <MetricCard :label="pick('实验数', 'Experiments')" :value="summary.total_experiments" />
      <MetricCard :label="pick('模型数', 'Models')" :value="summary.n_models" />
      <MetricCard :label="pick('划分协议', 'Split protocols')" :value="summary.n_split_protocols" />
      <MetricCard :label="pick('最佳 RMSE', 'Best RMSE')" :value="summary.best_rmse?.toFixed(3)" unit="evap anomaly" />
      <MetricCard :label="pick('最佳模型', 'Best model')" :value="summary.best_model" />
    </div>

    <div class="card p-5">
      <h2 class="text-lg font-semibold text-[var(--color-text)] mb-3">{{ pick('任务定义', 'Task definition') }}</h2>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div><span class="text-[var(--color-muted)]">{{ pick('输入窗口', 'Input') }}</span><p class="font-medium">{{ task.input_window }}</p></div>
        <div><span class="text-[var(--color-muted)]">{{ pick('目标变量', 'Target') }}</span><p class="font-medium">{{ task.target }}</p></div>
        <div><span class="text-[var(--color-muted)]">{{ pick('时间单位', 'Temporal unit') }}</span><p class="font-medium">{{ task.temporal_unit }}</p></div>
        <div><span class="text-[var(--color-muted)]">{{ pick('预测步长', 'Forecast horizon') }}</span><p class="font-medium">{{ task.forecast_horizon }}</p></div>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="card p-5">
        <h2 class="text-lg font-semibold text-[var(--color-text)] mb-3">{{ pick('评测区域', 'Benchmark regions') }} ({{ regions.length }})</h2>
        <table class="data-table">
          <thead><tr><th>{{ pick('区域', 'Region') }}</th><th>{{ pick('气候类型', 'Climate') }}</th><th>{{ pick('范围', 'Bounds') }}</th></tr></thead>
          <tbody>
            <tr v-for="r in regions" :key="r.name">
              <td class="font-medium">{{ r.name }}</td>
              <td><span class="badge badge-teal">{{ r.climate_type }}</span></td>
              <td class="text-xs text-gray-500">lat {{ r.lat_min }}-{{ r.lat_max }}, lon {{ r.lon_min }}-{{ r.lon_max }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="card p-5">
        <h2 class="text-lg font-semibold text-[var(--color-text)] mb-3">{{ pick('数据划分协议', 'Split protocols') }} ({{ splits.length }})</h2>
        <ul class="space-y-2 text-sm">
          <li v-for="s in splits" :key="s.split_id" class="flex items-start gap-2">
            <span class="badge badge-blue mt-0.5">{{ s.protocol }}</span>
            <span class="text-gray-600">{{ s.note?.substring(0, 80) }}...</span>
          </li>
        </ul>
      </div>
    </div>

    <div class="card p-5">
      <h2 class="text-lg font-semibold text-[var(--color-text)] mb-3">{{ pick('评测流程', 'Benchmark pipeline') }}</h2>
      <div class="flex flex-wrap gap-2 text-xs items-center">
        <span class="badge badge-purple">1. ERA5-Land Data</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">2. Feature Engineering</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">3. Forecasting Dataset</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">4. Split Protocols</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">5. Model Training</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">6. Evaluation</span><span class="text-gray-300">→</span>
        <span class="badge badge-cyan">7. Leaderboard</span>
      </div>
    </div>
  </div>
</template>
