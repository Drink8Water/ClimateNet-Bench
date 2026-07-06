<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  fetchEvaluationRun,
  fetchEvaluationRunStatus,
  fetchPlatformHealth,
  fetchPlatformLeaderboard,
  uploadSubmission,
} from '../api/climateApi'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'
import { useI18n } from '../i18n'

const health = ref(null)
const leaderboard = ref([])
const loading = ref(true)
const error = ref('')
const submitting = ref(false)
const submitResult = ref(null)
const submitError = ref('')
const runStatus = ref(null)
const runDetail = ref(null)
const polling = ref(false)
const selectedFile = ref(null)
const { pick } = useI18n()
const router = useRouter()

const form = ref({
  model_id: 1,
  benchmark_task_id: 1,
  split_protocol_id: 1,
  name: 'manual-submission',
})

const bestRun = computed(() => leaderboard.value[0] || null)
const latestRun = computed(() => runDetail.value || null)
const queueState = computed(() => {
  if (submitting.value) return pick('正在创建提交', 'Creating submission')
  if (polling.value) return pick('后台评测中', 'Worker evaluating')
  if (runStatus.value?.status === 'COMPLETED') return pick('最近运行完成', 'Latest run complete')
  if (runStatus.value?.status === 'FAILED') return pick('最近运行失败', 'Latest run failed')
  return pick('等待运行', 'Idle')
})
const primaryMetrics = computed(() => {
  const names = ['rmse', 'mae', 'r2', 'soil_moisture_drought_csi']
  return names
    .map(name => latestRun.value?.metrics?.find(metric => metric.name === name))
    .filter(Boolean)
})

async function loadPlatform() {
  loading.value = true
  error.value = ''
  try {
    const [h, lb] = await Promise.all([
      fetchPlatformHealth(),
      fetchPlatformLeaderboard(),
    ])
    health.value = h
    leaderboard.value = lb.results || []
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
  } finally {
    loading.value = false
  }
}

async function runDemoEvaluation() {
  const csv = [
    'actual,prediction,soil_moisture_drought,soil_moisture_drought_pred',
    '1.0,1.0,1,1',
    '2.0,2.4,1,0',
    '3.0,2.8,0,1',
    '4.0,4.1,0,0',
    '5.0,4.7,1,1',
    '6.0,6.2,0,0',
  ].join('\n')
  const file = new File([`${csv}\n`], 'demo_prediction.csv', { type: 'text/csv' })
  await createEvaluation(file, `auto-demo-${new Date().toISOString().slice(0, 19)}`)
}

async function submitPrediction() {
  if (!selectedFile.value) {
    submitError.value = pick('请选择 prediction.csv 文件。', 'Please choose a prediction.csv file.')
    return
  }

  await createEvaluation(selectedFile.value, form.value.name || 'manual-submission')
}

async function createEvaluation(predictionFile, submissionName) {
  submitting.value = true
  submitError.value = ''
  submitResult.value = null
  runStatus.value = null
  runDetail.value = null
  try {
    const formData = new FormData()
    formData.append('model_id', String(Number(form.value.model_id)))
    formData.append('benchmark_task_id', String(Number(form.value.benchmark_task_id)))
    formData.append('split_protocol_id', String(Number(form.value.split_protocol_id)))
    formData.append('name', submissionName)
    formData.append('prediction_csv', predictionFile)

    submitResult.value = await uploadSubmission(formData)
    await pollEvaluationRun(submitResult.value.evaluation_run_id)
  } catch (e) {
    submitError.value = e.response?.data?.detail || e.message
  } finally {
    submitting.value = false
  }
}

async function pollEvaluationRun(runId) {
  polling.value = true
  try {
    for (let attempt = 0; attempt < 150; attempt += 1) {
      runStatus.value = await fetchEvaluationRunStatus(runId)
      if (['COMPLETED', 'FAILED'].includes(runStatus.value.status)) break
      await new Promise(resolve => setTimeout(resolve, 2000))
    }
    runDetail.value = await fetchEvaluationRun(runId)
    await loadPlatform()
  } finally {
    polling.value = false
  }
}

function handleFileChange(event) {
  const [file] = event.target.files || []
  selectedFile.value = file || null
}

onMounted(loadPlatform)
</script>

<template>
  <div v-if="loading"><LoadingState :message="pick('正在加载评测平台...', 'Loading evaluation platform...')" /></div>
  <div v-else class="space-y-6">
    <section class="grid grid-cols-1 xl:grid-cols-[1.2fr_0.8fr] gap-5">
      <div class="card p-6">
        <div class="flex items-start justify-between gap-6">
          <div>
            <p class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('实验评测看板', 'experiment operations board') }}</p>
            <h1 class="mt-2 text-3xl font-bold text-[var(--color-text)]">{{ pick('自动评测控制台', 'Automated evaluation console') }}</h1>
            <p class="mt-2 max-w-2xl text-sm leading-6 text-[var(--color-muted)]">
              {{ pick('一键生成 demo prediction.csv，提交到后端队列，由 worker 计算指标并刷新数据库排行榜。手动上传保留给真实模型结果。', 'Generate a demo prediction.csv, submit it to the backend queue, let the worker compute metrics, and refresh the DB leaderboard. Manual upload remains available for real model outputs.') }}
            </p>
          </div>
          <span class="status-chip">{{ health?.status || 'unknown' }}</span>
        </div>

        <div class="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('队列状态', 'queue state') }}</div>
            <div class="mt-1 font-semibold">{{ queueState }}</div>
          </div>
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('最近任务', 'latest run') }}</div>
            <div class="mt-1 font-mono font-semibold">#{{ submitResult?.evaluation_run_id || '—' }}</div>
          </div>
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('排行榜行数', 'leaderboard rows') }}</div>
            <div class="mt-1 font-semibold">{{ leaderboard.length }}</div>
          </div>
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('最佳 RMSE', 'best RMSE') }}</div>
            <div class="mt-1 font-mono font-semibold">{{ bestRun?.rmse?.toFixed?.(4) ?? '—' }}</div>
          </div>
        </div>

        <div class="mt-6 flex flex-col gap-3 sm:flex-row">
          <button class="btn-primary" type="button" :disabled="submitting || polling" @click="runDemoEvaluation">
            {{ submitting || polling ? pick('自动运行中...', 'Running demo...') : pick('一键运行 demo evaluation', 'Run demo evaluation') }}
          </button>
          <button class="btn-secondary" type="button" @click="loadPlatform">{{ pick('刷新看板', 'Refresh board') }}</button>
        </div>

        <div v-if="submitResult" class="mt-6 space-y-3 rounded-lg bg-[var(--color-panel)] p-4 text-sm">
          <div class="flex items-center justify-between gap-3">
            <span>{{ pick('评测任务', 'Evaluation run') }} #{{ submitResult.evaluation_run_id }}</span>
            <span class="badge" :class="runStatus?.status === 'FAILED' ? 'badge-red' : 'badge-teal'">
              {{ runStatus?.status || submitResult.status }}
            </span>
          </div>
          <div class="h-2 overflow-hidden rounded-full bg-white">
            <div
              class="h-full rounded-full bg-[var(--color-accent)] transition-all"
              :style="{ width: `${runStatus?.progress_percent || 5}%` }"
            ></div>
          </div>
          <p class="text-xs text-[var(--color-muted)]">
            {{ polling ? pick('正在轮询评测状态...', 'Polling evaluation status...') : pick('评测状态已同步。', 'Evaluation status synchronized.') }}
          </p>
          <div v-if="primaryMetrics.length" class="grid grid-cols-2 gap-2 lg:grid-cols-4">
            <div v-for="metric in primaryMetrics" :key="metric.name" class="rounded-md bg-white p-3">
              <div class="text-[10px] text-[var(--color-muted)]">{{ metric.name }}</div>
              <div class="font-mono font-semibold">{{ metric.value?.toFixed?.(4) ?? metric.value }}</div>
            </div>
          </div>
          <button class="btn-secondary w-full" type="button" @click="router.push(`/evaluation/${submitResult.evaluation_run_id}`)">
            {{ pick('查看评测详情', 'View evaluation detail') }}
          </button>
        </div>
      </div>

      <form class="card p-5 space-y-4" @submit.prevent="submitPrediction">
        <div>
          <h2 class="text-lg font-semibold">{{ pick('手动提交真实预测', 'Manual prediction upload') }}</h2>
          <p class="mt-1 text-sm text-[var(--color-muted)]">{{ pick('用于上传真实模型产出的 prediction.csv。日常演示优先使用左侧的一键运行。', 'Use this for real model prediction CSVs. For demos, prefer the one-click run on the left.') }}</p>
        </div>

        <div class="grid grid-cols-3 gap-3">
          <label class="space-y-1 text-xs font-semibold text-[var(--color-muted)]">
            {{ pick('模型 ID', 'model id') }}
            <input v-model="form.model_id" type="number" min="1" />
          </label>
          <label class="space-y-1 text-xs font-semibold text-[var(--color-muted)]">
            {{ pick('任务 ID', 'task id') }}
            <input v-model="form.benchmark_task_id" type="number" min="1" />
          </label>
          <label class="space-y-1 text-xs font-semibold text-[var(--color-muted)]">
            {{ pick('划分 ID', 'split id') }}
            <input v-model="form.split_protocol_id" type="number" min="1" />
          </label>
        </div>

        <label class="block space-y-1 text-xs font-semibold text-[var(--color-muted)]">
          {{ pick('提交名称', 'submission name') }}
          <input v-model="form.name" class="w-full" />
        </label>
        <label class="block space-y-1 text-xs font-semibold text-[var(--color-muted)]">
          {{ pick('prediction.csv 文件', 'prediction.csv file') }}
          <input class="w-full" type="file" accept=".csv,text/csv" required @change="handleFileChange" />
        </label>
        <div v-if="selectedFile" class="rounded-md border border-[var(--color-border)] bg-white px-3 py-2 text-xs text-[var(--color-muted)]">
          <span class="font-semibold text-[var(--color-text)]">{{ selectedFile.name }}</span>
          <span class="ml-2">{{ (selectedFile.size / 1024).toFixed(1) }} KB</span>
        </div>

        <button class="btn-primary w-full" type="submit" :disabled="submitting">
          {{ submitting ? pick('提交中...', 'Submitting...') : pick('提交评测', 'Submit for evaluation') }}
        </button>

        <ErrorMessage v-if="submitError" :message="submitError" />
      </form>
    </section>

    <ErrorMessage v-if="error" :message="error" />

    <section class="card overflow-hidden">
      <div class="flex items-center justify-between border-b border-[var(--color-border)] px-5 py-4">
        <div>
          <h2 class="text-lg font-semibold">{{ pick('数据库排行榜', 'DB leaderboard') }}</h2>
          <p class="text-sm text-[var(--color-muted)]">{{ pick('已完成的评测任务按 RMSE 排名。', 'Completed evaluation runs ranked by RMSE.') }}</p>
        </div>
        <button class="btn-secondary" @click="loadPlatform">{{ pick('刷新', 'Refresh') }}</button>
      </div>
      <div v-if="!leaderboard.length" class="p-8 text-sm text-[var(--color-muted)]">
        {{ pick('还没有数据库评测结果。配置 DATABASE_URL，准备 model/task/split 记录后即可提交 prediction.csv。', 'No DB-backed evaluation results yet. Configure DATABASE_URL, seed model/task/split rows, then submit a prediction CSV.') }}
      </div>
      <table v-else class="data-table">
        <thead>
          <tr><th>{{ pick('排名', 'Rank') }}</th><th>{{ pick('模型', 'Model') }}</th><th>{{ pick('划分', 'Split') }}</th><th>RMSE</th><th>MAE</th><th>R2</th><th>{{ pick('任务', 'Run') }}</th></tr>
        </thead>
        <tbody>
          <tr v-for="row in leaderboard" :key="row.evaluation_run_id">
            <td class="font-bold">{{ row.rank }}</td>
            <td class="font-semibold">{{ row.model_name }}</td>
            <td><span class="badge badge-teal">{{ row.split_protocol }}</span></td>
            <td class="font-mono">{{ row.rmse?.toFixed?.(4) ?? '—' }}</td>
            <td class="font-mono">{{ row.mae?.toFixed?.(4) ?? '—' }}</td>
            <td class="font-mono">{{ row.r2?.toFixed?.(4) ?? '—' }}</td>
            <td>
              <button class="font-mono text-[var(--color-accent-strong)]" @click="router.push(`/evaluation/${row.evaluation_run_id}`)">
                #{{ row.evaluation_run_id }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>
