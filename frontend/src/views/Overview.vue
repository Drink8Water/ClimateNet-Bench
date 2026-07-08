<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { fetchBenchmarkRegions, fetchBenchmarkSplits, fetchBenchmarkSummary } from '../api/climateApi'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'
import { sampleBenchmark } from '../data/sampleBenchmark'
import { useI18n } from '../i18n'

const summary = ref(null)
const regions = ref([])
const splits = ref([])
const loading = ref(true)
const error = ref('')
const { pick } = useI18n()
const router = useRouter()

const best = computed(() => sampleBenchmark.leaderboard[0])
const topRows = computed(() => sampleBenchmark.leaderboard.slice(0, 5))
const bestBySplit = computed(() => {
  return sampleBenchmark.splits.map(split => {
    const rows = sampleBenchmark.leaderboard.filter(row => row.split_protocol === split)
    return rows.sort((a, b) => a.rmse - b.rmse)[0]
  })
})
const baselineGap = computed(() => {
  const climatology = sampleBenchmark.leaderboard.find(row => row.model_name === 'climatology' && row.split_protocol === best.value.split_protocol)
  if (!climatology) return null
  return 1 - best.value.rmse / climatology.rmse
})

onMounted(async () => {
  try {
    const [s, r, sp] = await Promise.all([
      fetchBenchmarkSummary().catch(() => null),
      fetchBenchmarkRegions().catch(() => []),
      fetchBenchmarkSplits().catch(() => []),
    ])
    summary.value = s
    regions.value = r.length ? r : sampleBenchmark.regions.map(name => ({ name, climate_type: 'sample region' }))
    splits.value = sp.length ? sp : sampleBenchmark.splits.map(protocol => ({ protocol, note: 'Fixed sample benchmark split' }))
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

function fmt(value, digits = 3) {
  return typeof value === 'number' ? value.toFixed(digits) : '—'
}
</script>

<template>
  <div v-if="loading"><LoadingState :message="pick('正在加载实验工作台...', 'Loading workbench...')" /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-6">
    <section class="page-hero p-6 lg:p-8">
      <div class="grid grid-cols-1 gap-6 xl:grid-cols-[1.25fr_0.75fr] xl:items-end">
        <div>
          <p class="text-xs font-semibold text-[var(--color-muted)]">
            {{ pick('可复现 benchmark + 本地分析工作台', 'reproducible benchmark + local analysis workbench') }}
          </p>
          <h1 class="mt-2 max-w-4xl text-3xl font-bold leading-tight text-[var(--color-text)] lg:text-4xl">
            {{ pick('ERA5-Land 水文气候胁迫预测基准', 'ERA5-Land hydroclimate stress benchmark') }}
          </h1>
          <p class="mt-4 max-w-3xl text-sm leading-6 text-[var(--color-muted)]">
            {{ pick('项目核心不是通用上传平台，而是固定数据来源、固定评测协议和统一指标下的多模型泛化比较。前端用于解释结果：哪个模型更稳，在哪类 split 失效，极端事件指标是否可信。', 'This is not a generic upload platform. It is a fixed-source benchmark with fixed evaluation protocols and unified metrics. The dashboard explains which models generalize, where they fail, and how event metrics behave.') }}
          </p>
          <div class="mt-5 flex flex-wrap gap-2">
            <span class="badge badge-teal">{{ sampleBenchmark.source }}</span>
            <span class="badge badge-blue">{{ sampleBenchmark.target }}</span>
            <span class="badge badge-purple">{{ sampleBenchmark.event }}</span>
          </div>
        </div>
        <div class="analysis-panel p-5">
          <p class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('sample bench 结果', 'sample bench result') }}</p>
          <div class="mt-3 flex items-baseline gap-2">
            <span class="text-4xl font-bold text-[var(--color-text)]">{{ fmt(best.rmse, 3) }}</span>
            <span class="text-sm text-[var(--color-muted)]">RMSE</span>
          </div>
          <p class="mt-2 text-sm text-[var(--color-muted)]">
            {{ best.model_name }} · {{ best.split_protocol }} · R2 {{ fmt(best.r2, 2) }}
          </p>
          <button class="btn-primary mt-5 w-full" type="button" @click="router.push('/leaderboard')">
            {{ pick('查看完整排行榜', 'Open leaderboard') }}
          </button>
        </div>
      </div>
    </section>

    <section class="grid grid-cols-2 gap-4 lg:grid-cols-5">
      <div class="card p-4">
        <span class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('样本数', 'samples') }}</span>
        <div class="mt-1 text-2xl font-bold">{{ sampleBenchmark.nSamples }}</div>
      </div>
      <div class="card p-4">
        <span class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('模型', 'models') }}</span>
        <div class="mt-1 text-2xl font-bold">{{ sampleBenchmark.models.length }}</div>
      </div>
      <div class="card p-4">
        <span class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('划分协议', 'splits') }}</span>
        <div class="mt-1 text-2xl font-bold">{{ sampleBenchmark.splits.length }}</div>
      </div>
      <div class="card p-4">
        <span class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('气候区', 'climate zones') }}</span>
        <div class="mt-1 text-2xl font-bold">{{ sampleBenchmark.climateZones.length }}</div>
      </div>
      <div class="card p-4">
        <span class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('相对气候态提升', 'skill vs climatology') }}</span>
        <div class="mt-1 text-2xl font-bold">{{ baselineGap == null ? '—' : `${fmt(baselineGap * 100, 1)}%` }}</div>
      </div>
    </section>

    <section class="grid grid-cols-1 gap-5 xl:grid-cols-[0.7fr_0.3fr]">
      <div class="card overflow-hidden">
        <div class="border-b border-[var(--color-border)] px-5 py-4">
          <h2 class="text-lg font-semibold">{{ pick('sample benchmark 排行榜', 'sample benchmark leaderboard') }}</h2>
          <p class="mt-1 text-sm text-[var(--color-muted)]">
            {{ pick('由 scripts/demo_smoke.py 生成：3 个模型 × 3 个 split，适合 README 截图和快速复现。', 'Generated by scripts/demo_smoke.py: 3 models x 3 splits, suitable for README screenshots and quick reproduction.') }}
          </p>
        </div>
        <table class="data-table">
          <thead>
            <tr>
              <th>{{ pick('排名', 'Rank') }}</th>
              <th>{{ pick('模型', 'Model') }}</th>
              <th>{{ pick('划分', 'Split') }}</th>
              <th>RMSE</th>
              <th>R2</th>
              <th>CSI</th>
              <th>POD</th>
              <th>FAR</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in topRows" :key="`${row.model_name}-${row.split_protocol}`">
              <td class="font-bold">{{ row.rank }}</td>
              <td class="font-semibold">{{ row.model_name }}</td>
              <td><span class="badge badge-teal">{{ row.split_protocol }}</span></td>
              <td class="font-mono">{{ fmt(row.rmse, 3) }}</td>
              <td class="font-mono">{{ fmt(row.r2, 2) }}</td>
              <td class="font-mono">{{ fmt(row.soil_moisture_drought_csi, 2) }}</td>
              <td class="font-mono">{{ fmt(row.soil_moisture_drought_pod, 2) }}</td>
              <td class="font-mono">{{ fmt(row.soil_moisture_drought_far, 2) }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <aside class="space-y-4">
        <div class="card p-5">
          <h2 class="text-lg font-semibold">{{ pick('数据来源边界', 'data boundary') }}</h2>
          <p class="mt-2 text-sm leading-6 text-[var(--color-muted)]">
            {{ pick('sample bench 使用合成的 ERA5-Land 风格数据，只用于演示工作流。正式实验应使用预处理后的 ERA5-Land 数据，保持固定任务和固定 split，避免变成任意上传数据的平台。', 'The sample bench uses synthetic ERA5-Land-style data for workflow demonstration. Scientific runs should use processed ERA5-Land data with fixed tasks and fixed splits, not arbitrary uploaded datasets.') }}
          </p>
        </div>
        <div class="card p-5">
          <h2 class="text-lg font-semibold">{{ pick('主流程', 'primary workflow') }}</h2>
          <ol class="mt-3 space-y-3 text-sm text-[var(--color-muted)]">
            <li>1. {{ pick('运行 sample bench 或正式 benchmark suite', 'Run sample bench or the full benchmark suite') }}</li>
            <li>2. {{ pick('生成 leaderboard / predictions / run summary', 'Generate leaderboard, predictions, and run summary') }}</li>
            <li>3. {{ pick('用工作台解释模型排名和泛化失败', 'Use the workbench to diagnose model ranking and failures') }}</li>
          </ol>
        </div>
      </aside>
    </section>

    <section class="grid grid-cols-1 gap-5 lg:grid-cols-3">
      <article v-for="row in bestBySplit" :key="row.split_protocol" class="card p-5">
        <p class="text-xs font-semibold text-[var(--color-muted)]">{{ row.split_protocol }}</p>
        <h3 class="mt-2 text-xl font-bold">{{ row.model_name }}</h3>
        <div class="mt-4 grid grid-cols-3 gap-3 text-sm">
          <div>
            <div class="text-xs text-[var(--color-muted)]">RMSE</div>
            <div class="font-mono font-semibold">{{ fmt(row.rmse, 3) }}</div>
          </div>
          <div>
            <div class="text-xs text-[var(--color-muted)]">R2</div>
            <div class="font-mono font-semibold">{{ fmt(row.r2, 2) }}</div>
          </div>
          <div>
            <div class="text-xs text-[var(--color-muted)]">CSI</div>
            <div class="font-mono font-semibold">{{ fmt(row.soil_moisture_drought_csi, 2) }}</div>
          </div>
        </div>
      </article>
    </section>

    <section class="grid grid-cols-1 gap-5 xl:grid-cols-2">
      <div class="card p-5">
        <h2 class="text-lg font-semibold">{{ pick('评测区域', 'benchmark regions') }}</h2>
        <div class="mt-4 flex flex-wrap gap-2">
          <span v-for="region in regions" :key="region.name" class="badge badge-blue">
            {{ region.name }}
          </span>
        </div>
      </div>
      <div class="card p-5">
        <h2 class="text-lg font-semibold">{{ pick('固定 split 协议', 'fixed split protocols') }}</h2>
        <div class="mt-4 flex flex-wrap gap-2">
          <span v-for="split in splits" :key="split.protocol" class="badge badge-purple">
            {{ split.protocol }}
          </span>
        </div>
      </div>
    </section>

    <section class="card p-5">
      <h2 class="text-lg font-semibold">{{ pick('复现命令', 'reproduction command') }}</h2>
      <pre class="mt-3 overflow-x-auto rounded-md bg-[var(--color-panel)] p-4 text-xs text-[var(--color-text)]">PYTHONPATH=src python scripts/demo_smoke.py</pre>
      <p class="mt-3 text-sm text-[var(--color-muted)]">
        {{ pick('这个命令生成可截图的 sample benchmark 产物；后续 README 应展示这张看板和 leaderboard。', 'This command generates screenshot-ready sample benchmark artifacts. The README should show this dashboard and leaderboard.') }}
      </p>
    </section>
  </div>
</template>
