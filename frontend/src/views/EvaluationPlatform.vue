<script setup>
import { onMounted, ref } from 'vue'
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

async function submitPrediction() {
  if (!selectedFile.value) {
    submitError.value = pick('请选择 prediction.csv 文件。', 'Please choose a prediction.csv file.')
    return
  }

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
    formData.append('name', form.value.name || 'manual-submission')
    formData.append('prediction_csv', selectedFile.value)

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
    <section class="grid grid-cols-1 xl:grid-cols-[1.1fr_0.9fr] gap-5">
      <div class="card p-6">
        <div class="flex items-start justify-between gap-6">
          <div>
            <p class="text-xs font-semibold text-[var(--color-muted)]">{{ pick('后端评测平台', 'backend platform') }}</p>
            <h1 class="mt-2 text-3xl font-bold text-[var(--color-text)]">{{ pick('提交评测队列', 'Submission evaluation queue') }}</h1>
            <p class="mt-2 max-w-2xl text-sm leading-6 text-[var(--color-muted)]">
              {{ pick('登记 prediction.csv，保存为本地 artifact，并由后台 worker 计算指标写入数据库排行榜。', 'Register a prediction CSV, store it as a local artifact, and let the worker compute metrics for the DB leaderboard.') }}
            </p>
          </div>
          <span class="status-chip">{{ health?.status || 'unknown' }}</span>
        </div>

        <div class="mt-6 grid grid-cols-3 gap-3">
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('artifact 存储', 'artifact mode') }}</div>
            <div class="mt-1 font-semibold">{{ pick('本地文件系统', 'local filesystem') }}</div>
          </div>
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('评测方式', 'evaluation') }}</div>
            <div class="mt-1 font-semibold">{{ pick('异步任务', 'Celery async') }}</div>
          </div>
          <div class="rounded-lg bg-[var(--color-panel)] p-4">
            <div class="text-xs text-[var(--color-muted)]">{{ pick('排行榜行数', 'leaderboard rows') }}</div>
            <div class="mt-1 font-semibold">{{ leaderboard.length }}</div>
          </div>
        </div>
      </div>

      <form class="card p-5 space-y-4" @submit.prevent="submitPrediction">
        <div>
          <h2 class="text-lg font-semibold">{{ pick('创建提交', 'Create submission') }}</h2>
          <p class="mt-1 text-sm text-[var(--color-muted)]">{{ pick('上传 prediction.csv，文件需包含 actual 和 prediction 两列。', 'Upload a prediction.csv file with actual and prediction columns.') }}</p>
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
        <div v-if="submitResult" class="space-y-3 rounded-lg bg-[var(--color-panel)] p-4 text-sm">
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
          <div v-if="runDetail?.metrics?.length" class="grid grid-cols-3 gap-2">
            <div v-for="metric in runDetail.metrics" :key="metric.name" class="rounded-md bg-white p-2">
              <div class="text-[10px] text-[var(--color-muted)]">{{ metric.name }}</div>
              <div class="font-mono font-semibold">{{ metric.value?.toFixed?.(4) ?? metric.value }}</div>
            </div>
          </div>
          <button class="btn-secondary w-full" type="button" @click="router.push(`/evaluation/${submitResult.evaluation_run_id}`)">
            {{ pick('查看评测详情', 'View evaluation detail') }}
          </button>
        </div>
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
