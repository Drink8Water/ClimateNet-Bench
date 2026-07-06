<script setup>
import { computed, ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { fetchLeaderboard, fetchBenchmarkSummary, fetchPlatformLeaderboard } from '../api/climateApi'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'
import { useI18n } from '../i18n'

const rows = ref([])
const meta = ref(null)
const loading = ref(true)
const error = ref('')
const usingFallback = ref(false)
const splitFilter = ref('')
const modelFilter = ref('')
const eventFilter = ref('')
const sortKey = ref('rmse')
const { pick } = useI18n()
const router = useRouter()

async function loadLeaderboard() {
  loading.value = true
  error.value = ''
  usingFallback.value = false
  try {
    const [lb, m] = await Promise.all([
      fetchPlatformLeaderboard({
        metric: sortKey.value,
        split_protocol: splitFilter.value || undefined,
        event_type: eventFilter.value || undefined,
      }),
      fetchBenchmarkSummary().catch(() => null),
    ])
    rows.value = normalizeRows(lb.results || [])
    meta.value = m
  } catch (platformError) {
    try {
      const [lb, m] = await Promise.all([fetchLeaderboard({ limit: 500 }), fetchBenchmarkSummary()])
      rows.value = normalizeRows(lb)
      meta.value = m
      usingFallback.value = true
    } catch (fallbackError) {
      error.value = fallbackError.response?.data?.detail || fallbackError.message || platformError.message
      rows.value = []
    }
  } finally {
    loading.value = false
  }
}

onMounted(loadLeaderboard)

const filtered = computed(() => {
  let r = rows.value
  if (modelFilter.value) r = r.filter(x => x.model_name === modelFilter.value)
  return [...r].sort((a, b) => (a[sortKey.value] ?? Infinity) - (b[sortKey.value] ?? Infinity))
})

const topRows = computed(() => filtered.value.slice(0, 3))
const bestOverall = computed(() => rows.value.reduce((best, row) => {
  if (!best) return row
  return (row.rmse ?? Infinity) < (best.rmse ?? Infinity) ? row : best
}, null))

const unique = (key) => [...new Set(rows.value.map(r => r[key]))].sort()
const hasEventMetrics = computed(() => rows.value.some(r => r.soil_moisture_drought_csi != null))

function fmt(v) { return typeof v === 'number' ? v.toFixed(4) : v || '—' }
function bestInSplit(split) {
  const inSplit = rows.value.filter(r => r.split_protocol === split)
  if (!inSplit.length) return null
  return inSplit.reduce((a, b) => (a.rmse || 99) < (b.rmse || 99) ? a : b)
}

function normalizeRows(inputRows) {
  return inputRows.map((row, index) => ({
    ...row,
    experiment_id: row.experiment_id || row.evaluation_run_id || `row-${index}`,
    rank: row.rank || index + 1,
  }))
}
</script>

<template>
  <div v-if="loading"><LoadingState :message="pick('正在加载排行榜...', 'Loading leaderboard...')" /></div>
  <div v-else-if="error" class="space-y-4">
    <ErrorMessage :message="error" />
    <button class="btn-secondary" type="button" @click="loadLeaderboard">{{ pick('重试', 'Retry') }}</button>
  </div>
  <div v-else class="space-y-6">
    <section class="grid grid-cols-1 xl:grid-cols-[1.35fr_0.65fr] gap-5">
      <div class="card p-6">
        <p class="text-xs font-semibold text-[var(--color-muted)]">ranked benchmark results</p>
        <div class="mt-2 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h1 class="text-3xl font-bold text-[var(--color-text)]">{{ pick('模型排行榜', 'Leaderboard') }}</h1>
            <p class="mt-2 max-w-2xl text-sm leading-6 text-[var(--color-muted)]">
              {{ pick('比较数据库评测结果在不同划分协议和事件指标下的泛化表现。默认按 RMSE 从低到高排序。', 'Compare DB-backed evaluation results across split protocols and event metrics. Lower RMSE ranks higher by default.') }}
            </p>
            <p v-if="usingFallback" class="mt-2 text-xs text-[#8a6b2f]">
              {{ pick('当前后端平台接口不可用，已切换到旧分析数据。', 'Platform API unavailable; showing legacy analysis data.') }}
            </p>
          </div>
          <div class="grid grid-cols-2 gap-3 text-sm">
            <div class="rounded-lg bg-[var(--color-panel)] px-4 py-3">
              <div class="text-xs text-[var(--color-muted)]">{{ pick('结果行', 'Rows') }}</div>
              <div class="mt-1 text-xl font-bold">{{ rows.length }}</div>
            </div>
            <div class="rounded-lg bg-[var(--color-panel)] px-4 py-3">
              <div class="text-xs text-[var(--color-muted)]">{{ pick('最佳 RMSE', 'Best RMSE') }}</div>
              <div class="mt-1 text-xl font-bold">{{ fmt(bestOverall?.rmse) }}</div>
            </div>
          </div>
        </div>
      </div>

      <div class="card p-5">
        <p class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('当前领先模型', 'Current leader') }}</p>
        <div class="mt-3 text-2xl font-bold">{{ bestOverall?.model_name || '—' }}</div>
        <div class="mt-3 flex flex-wrap gap-2">
          <span class="badge badge-teal">{{ bestOverall?.split_protocol || 'no split' }}</span>
          <span class="badge badge-purple">RMSE {{ fmt(bestOverall?.rmse) }}</span>
        </div>
        <p class="mt-4 text-sm text-[var(--color-muted)]">{{ pick('基于当前所有可用基准结果的单次最佳运行。', 'Best single run across all available benchmark rows.') }}</p>
      </div>
    </section>

    <section class="card p-4">
      <div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div class="flex gap-3 flex-wrap">
          <select v-model="splitFilter" @change="loadLeaderboard">
            <option value="">{{ pick('全部划分协议', 'All split protocols') }}</option>
            <option v-for="s in unique('split_protocol')" :key="s" :value="s">{{ s }}</option>
          </select>
          <select v-model="modelFilter">
            <option value="">{{ pick('全部模型', 'All models') }}</option>
            <option v-for="m in unique('model_name')" :key="m" :value="m">{{ m }}</option>
          </select>
          <select v-model="eventFilter" @change="loadLeaderboard">
            <option value="">{{ pick('全部事件类型', 'All event types') }}</option>
            <option value="soil_moisture_drought">{{ pick('土壤湿度干旱', 'Soil moisture drought') }}</option>
          </select>
        </div>
        <div class="flex items-center gap-3">
          <select v-model="sortKey" @change="loadLeaderboard">
            <option value="rmse">{{ pick('按 RMSE 排序', 'Sort by RMSE') }}</option>
            <option value="mae">{{ pick('按 MAE 排序', 'Sort by MAE') }}</option>
            <option value="soil_moisture_drought_csi">CSI</option>
            <option value="soil_moisture_drought_pod">POD</option>
            <option value="soil_moisture_drought_far">FAR</option>
          </select>
          <p class="text-xs text-[var(--color-muted)]">{{ filtered.length }} / {{ rows.length }}</p>
          <button class="btn-secondary" type="button" @click="loadLeaderboard">{{ pick('刷新', 'Refresh') }}</button>
        </div>
      </div>
    </section>

    <section v-if="!rows.length" class="card p-8 text-sm text-[var(--color-muted)]">
      {{ pick('还没有可展示的评测结果。请先在“评测平台”上传 prediction.csv，worker 完成后这里会出现排名。', 'No evaluation results yet. Upload a prediction.csv in the evaluation platform; rankings appear here after the worker completes.') }}
    </section>

    <section v-if="rows.length" class="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <article v-for="r in topRows" :key="`top-${r.experiment_id}`" class="card p-4">
        <div class="flex items-start justify-between gap-3">
          <div>
            <div class="text-xs text-[var(--color-muted)]">{{ pick('排名', 'Rank') }} {{ r.rank }}</div>
            <h2 class="mt-1 text-lg font-semibold">{{ r.model_name }}</h2>
          </div>
          <span class="badge badge-teal">{{ r.split_protocol }}</span>
        </div>
        <div class="mt-5 grid grid-cols-3 gap-2 text-sm">
          <div><div class="text-xs text-[var(--color-muted)]">RMSE</div><div class="font-mono font-semibold">{{ fmt(r.rmse) }}</div></div>
          <div><div class="text-xs text-[var(--color-muted)]">MAE</div><div class="font-mono font-semibold">{{ fmt(r.mae) }}</div></div>
          <div><div class="text-xs text-[var(--color-muted)]">R2</div><div class="font-mono font-semibold">{{ fmt(r.r2) }}</div></div>
        </div>
      </article>
    </section>

    <section v-if="rows.length" class="grid grid-cols-1 xl:grid-cols-[0.35fr_0.65fr] gap-5">
      <div class="card p-5">
        <h2 class="text-lg font-semibold">{{ pick('各划分最佳结果', 'Best per split') }}</h2>
        <div class="mt-4 space-y-3">
          <div v-for="s in unique('split_protocol')" :key="s" class="rounded-lg bg-[var(--color-panel)] p-3">
            <div class="flex items-center justify-between gap-3">
              <span class="badge badge-blue">{{ s }}</span>
              <span class="font-mono text-xs">RMSE {{ bestInSplit(s)?.rmse?.toFixed(3) }}</span>
            </div>
            <div class="mt-2 font-semibold">{{ bestInSplit(s)?.model_name }}</div>
          </div>
        </div>
      </div>

      <div class="card overflow-x-auto">
        <table class="data-table">
          <thead>
          <tr><th>{{ pick('排名', 'Rank') }}</th><th>{{ pick('模型', 'Model') }}</th><th>{{ pick('划分', 'Split') }}</th><th>{{ pick('特征集', 'Feature set') }}</th><th>RMSE ↓</th><th>MAE</th><th>R²</th><th v-if="hasEventMetrics">CSI</th><th v-if="hasEventMetrics">POD</th><th v-if="hasEventMetrics">FAR</th><th>{{ pick('相对持久性基线', 'Skill vs persistence') }}</th><th>{{ pick('详情', 'Detail') }}</th></tr>
          </thead>
          <tbody>
            <tr v-for="r in filtered" :key="r.experiment_id" :class="{ 'bg-[var(--color-panel)]': r.rank === 1 }">
              <td class="font-bold">{{ r.rank }}</td>
              <td class="font-semibold">{{ r.model_name }}</td>
              <td><span class="badge badge-teal">{{ r.split_protocol }}</span></td>
              <td>{{ r.feature_set || 'base' }}</td>
              <td class="font-mono font-semibold">{{ fmt(r.rmse) }}</td>
              <td class="font-mono text-[var(--color-muted)]">{{ fmt(r.mae) }}</td>
              <td class="font-mono" :class="r.r2 > 0 ? 'text-[var(--color-accent-strong)]' : 'text-[var(--color-muted)]'">{{ fmt(r.r2) }}</td>
              <td v-if="hasEventMetrics" class="font-mono">{{ fmt(r.soil_moisture_drought_csi) }}</td>
              <td v-if="hasEventMetrics" class="font-mono">{{ fmt(r.soil_moisture_drought_pod) }}</td>
              <td v-if="hasEventMetrics" class="font-mono">{{ fmt(r.soil_moisture_drought_far) }}</td>
              <td class="font-mono" :class="(r.skill_vs_persistence || 0) > 0 ? 'text-[var(--color-accent-strong)]' : 'text-[var(--color-danger)]'">{{ r.skill_vs_persistence != null ? (r.skill_vs_persistence * 100).toFixed(1) + '%' : '—' }}</td>
              <td>
                <button
                  v-if="r.evaluation_run_id"
                  class="font-mono text-[var(--color-accent-strong)]"
                  @click="router.push(`/evaluation/${r.evaluation_run_id}`)"
                >
                  #{{ r.evaluation_run_id }}
                </button>
                <span v-else class="text-[var(--color-muted)]">—</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>
