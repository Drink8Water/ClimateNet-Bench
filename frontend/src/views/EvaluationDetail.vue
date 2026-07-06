<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { fetchEvaluationRun, fetchEvaluationRunStatus } from '../api/climateApi'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'
import { useI18n } from '../i18n'

const route = useRoute()
const router = useRouter()
const { pick } = useI18n()

const loading = ref(true)
const error = ref('')
const detail = ref(null)
const status = ref(null)

const runId = computed(() => Number(route.params.id))
const metrics = computed(() => detail.value?.metrics || [])
const artifacts = computed(() => detail.value?.artifacts || [])
const metricMap = computed(() => Object.fromEntries(metrics.value.map(metric => [metric.name, metric.value])))

function fmt(value) {
  return typeof value === 'number' ? value.toFixed(4) : value ?? '—'
}

function statusClass(value) {
  if (value === 'FAILED') return 'badge-red'
  if (value === 'COMPLETED') return 'badge-teal'
  return 'badge-amber'
}

async function loadDetail() {
  loading.value = true
  error.value = ''
  try {
    const [nextStatus, nextDetail] = await Promise.all([
      fetchEvaluationRunStatus(runId.value),
      fetchEvaluationRun(runId.value),
    ])
    status.value = nextStatus
    detail.value = nextDetail
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
  } finally {
    loading.value = false
  }
}

onMounted(loadDetail)
</script>

<template>
  <div v-if="loading">
    <LoadingState :message="pick('正在加载评测详情...', 'Loading evaluation detail...')" />
  </div>
  <div v-else-if="error" class="space-y-4">
    <ErrorMessage :message="error" />
    <button class="btn-secondary" @click="router.push('/evaluation')">{{ pick('返回提交评测', 'Back to evaluation') }}</button>
  </div>
  <div v-else class="space-y-6">
    <section class="grid grid-cols-1 xl:grid-cols-[1.2fr_0.8fr] gap-5">
      <div class="card p-6">
        <div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('评测运行详情', 'evaluation run detail') }}</p>
            <h1 class="mt-2 text-3xl font-bold text-[var(--color-text)]">Run #{{ detail.evaluation_run_id }}</h1>
            <p class="mt-2 max-w-2xl text-sm leading-6 text-[var(--color-muted)]">
              {{ pick('这里汇总一次提交的状态、指标和产物，方便复盘实验结果。', 'Review the status, metrics, and artifacts produced by a single submission evaluation.') }}
            </p>
          </div>
          <span class="badge" :class="statusClass(detail.status)">{{ detail.status }}</span>
        </div>

        <div class="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('提交 ID', 'Submission ID') }}</div>
            <div class="mt-1 font-mono text-lg font-semibold">#{{ detail.submission_id }}</div>
          </div>
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('划分协议 ID', 'Split protocol ID') }}</div>
            <div class="mt-1 font-mono text-lg font-semibold">#{{ detail.split_protocol_id }}</div>
          </div>
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('进度', 'Progress') }}</div>
            <div class="mt-1 font-mono text-lg font-semibold">{{ status?.progress_percent ?? 0 }}%</div>
          </div>
        </div>

        <div class="mt-5 h-2 overflow-hidden rounded-full bg-[var(--color-panel)]">
          <div
            class="h-full rounded-full bg-[var(--color-accent)] transition-all"
            :style="{ width: `${status?.progress_percent || 0}%` }"
          ></div>
        </div>
      </div>

      <div class="card p-5">
        <h2 class="text-lg font-semibold">{{ pick('核心回归指标', 'Core regression metrics') }}</h2>
        <div class="mt-4 grid grid-cols-3 gap-3">
          <div class="rounded-lg bg-[var(--color-panel)] p-3">
            <div class="text-xs text-[var(--color-muted)]">RMSE</div>
            <div class="mt-1 font-mono text-xl font-bold">{{ fmt(metricMap.rmse) }}</div>
          </div>
          <div class="rounded-lg bg-[var(--color-panel)] p-3">
            <div class="text-xs text-[var(--color-muted)]">MAE</div>
            <div class="mt-1 font-mono text-xl font-bold">{{ fmt(metricMap.mae) }}</div>
          </div>
          <div class="rounded-lg bg-[var(--color-panel)] p-3">
            <div class="text-xs text-[var(--color-muted)]">R2</div>
            <div class="mt-1 font-mono text-xl font-bold">{{ fmt(metricMap.r2) }}</div>
          </div>
        </div>
        <p class="mt-4 text-sm leading-6 text-[var(--color-muted)]">
          {{ pick('RMSE 越低表示整体误差越小；R2 越高表示模型解释力越强。', 'Lower RMSE means smaller overall error; higher R2 indicates stronger explained variance.') }}
        </p>
      </div>
    </section>

    <section class="grid grid-cols-1 xl:grid-cols-[0.55fr_0.45fr] gap-5">
      <div class="card overflow-hidden">
        <div class="border-b border-[var(--color-border)] px-5 py-4">
          <h2 class="text-lg font-semibold">{{ pick('全部指标', 'All metrics') }}</h2>
        </div>
        <div v-if="!metrics.length" class="p-6 text-sm text-[var(--color-muted)]">
          {{ pick('当前评测还没有指标。', 'No metrics are available for this run yet.') }}
        </div>
        <table v-else class="data-table">
          <thead>
            <tr><th>{{ pick('指标', 'Metric') }}</th><th>{{ pick('数值', 'Value') }}</th><th>{{ pick('单位', 'Unit') }}</th></tr>
          </thead>
          <tbody>
            <tr v-for="metric in metrics" :key="metric.name">
              <td class="font-semibold">{{ metric.name }}</td>
              <td class="font-mono">{{ fmt(metric.value) }}</td>
              <td class="text-[var(--color-muted)]">{{ metric.unit || '—' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="card overflow-hidden">
        <div class="border-b border-[var(--color-border)] px-5 py-4">
          <h2 class="text-lg font-semibold">{{ pick('产物', 'Artifacts') }}</h2>
        </div>
        <div v-if="!artifacts.length" class="p-6 text-sm text-[var(--color-muted)]">
          {{ pick('当前评测没有关联产物。', 'No artifacts are linked to this run.') }}
        </div>
        <div v-else class="divide-y divide-[var(--color-border-soft)]">
          <div v-for="artifact in artifacts" :key="artifact.artifact_id" class="px-5 py-4">
            <div class="flex items-center justify-between gap-3">
              <span class="badge badge-blue">{{ artifact.artifact_type }}</span>
              <span class="font-mono text-xs text-[var(--color-muted)]">#{{ artifact.artifact_id }}</span>
            </div>
            <div class="mt-2 break-all font-mono text-xs text-[var(--color-muted)]">{{ artifact.uri }}</div>
          </div>
        </div>
      </div>
    </section>

    <ErrorMessage v-if="detail.error_message" :message="detail.error_message" />

    <div class="flex flex-wrap gap-3">
      <button class="btn-secondary" @click="router.push('/evaluation')">{{ pick('返回提交评测', 'Back to evaluation') }}</button>
      <button class="btn-secondary" @click="router.push('/leaderboard')">{{ pick('查看排行榜', 'Open leaderboard') }}</button>
      <button class="btn-primary" @click="loadDetail">{{ pick('刷新详情', 'Refresh detail') }}</button>
    </div>
  </div>
</template>
