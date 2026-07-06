import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  timeout: 15000,
})

const platformClient = axios.create({
  baseURL: '/platform-api',
  timeout: 15000,
})

async function get(path, params = {}) {
  const { data } = await client.get(path, { params })
  return data
}

async function platformGet(path, params = {}) {
  const { data } = await platformClient.get(path, { params })
  return data
}

async function platformPost(path, payload = {}) {
  const { data } = await platformClient.post(path, payload)
  return data
}

async function platformPostForm(path, formData) {
  const { data } = await platformClient.post(path, formData)
  return data
}

// ── Benchmark info ────────────────────────────────────────────────
export function fetchBenchmarkSummary() { return get('/benchmark/summary') }
export function fetchBenchmarkTask() { return get('/benchmark/task') }
export function fetchBenchmarkRegions() { return get('/benchmark/regions') }
export function fetchBenchmarkSplits() { return get('/benchmark/splits') }

// ── Leaderboard ────────────────────────────────────────────────────
export function fetchLeaderboard(params = {}) { return get('/leaderboard', params) }
export function fetchSplitDifficulty() { return get('/split-difficulty') }
export function fetchAblationStudy(params = {}) { return get('/ablation-study', params) }

// ── Experiments ────────────────────────────────────────────────────
export function fetchExperiments(params = {}) { return get('/experiments', params) }
export function fetchExperimentDetail(experimentId) { return get(`/experiments/${experimentId}`) }
export function fetchPredictions(experimentId, params = {}) { return get(`/experiments/${experimentId}/predictions`, params) }
export function fetchIntervals(experimentId, params = {}) { return get(`/experiments/${experimentId}/intervals`, params) }
export function fetchFeatureImportance(experimentId, params = {}) { return get(`/experiments/${experimentId}/feature-importance`, params) }
export function fetchModelComparison(params = {}) { return get('/model-comparison', params) }

// ── Uncertainty ────────────────────────────────────────────────────
export function fetchCalibration(params = {}) { return get('/uncertainty/calibration', params) }

// ── Physical consistency ───────────────────────────────────────────
export function fetchPhysicalSummary() { return get('/physical-consistency/summary') }
export function fetchRegionalSensitivity(params = {}) { return get('/physical-consistency/regional-sensitivity', params) }

// ── Spatial ────────────────────────────────────────────────────────
export function fetchSpatialGrid(params = {}) { return get('/spatial-grid', params) }
export function fetchTimeseries(params = {}) { return get('/timeseries', params) }

// ── Legacy / misc ──────────────────────────────────────────────────
export function fetchProjectSummary() { return get('/project-summary') }
export function fetchDatasetSummary() { return get('/dataset-summary') }

// ── Backend platform API ───────────────────────────────────────────
export function fetchPlatformHealth() { return platformGet('/health') }
export function fetchPlatformLeaderboard(params = {}) { return platformGet('/leaderboard', params) }
export function fetchEvaluationRun(runId) { return platformGet(`/evaluation-runs/${runId}`) }
export function fetchEvaluationRunStatus(runId) { return platformGet(`/evaluation-runs/${runId}/status`) }
export function createSubmission(payload) { return platformPost('/submissions', payload) }
export function uploadSubmission(formData) { return platformPostForm('/submissions/upload', formData) }
