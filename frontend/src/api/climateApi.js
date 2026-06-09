import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  timeout: 10000,
})

// ── Helpers ──────────────────────────────────────────────────────────
async function get(path, params = {}) {
  const { data } = await client.get(path, { params })
  return data
}

// ── Summary ───────────────────────────────────────────────────────────
export function fetchProjectSummary() {
  return get('/project-summary')
}
export function fetchDatasetSummary() {
  return get('/dataset-summary')
}

// ── Experiments ───────────────────────────────────────────────────────
export function fetchExperiments(params = {}) {
  return get('/experiments', params)
}
export function fetchExperimentSummary() {
  return get('/experiments/summary')
}
export function fetchExperimentDetail(experimentId) {
  return get(`/experiments/${experimentId}`)
}

// ── Comparison ────────────────────────────────────────────────────────
export function fetchModelComparison(params = {}) {
  return get('/model-comparison', params)
}
export function fetchAblationStudy(experimentId = 'latest') {
  return get('/ablation-study', { experiment_id: experimentId })
}

// ── Predictions ───────────────────────────────────────────────────────
export function fetchPredictions(experimentId, params = {}) {
  return get(`/experiments/${experimentId}/predictions`, params)
}
export function fetchResiduals(experimentId, params = {}) {
  return get(`/experiments/${experimentId}/residuals`, params)
}
export function fetchPredictionSummary(experimentId, params = {}) {
  return get(`/experiments/${experimentId}/prediction-summary`, params)
}

// ── Attribution ───────────────────────────────────────────────────────
export function fetchFeatureImportance(experimentId, params = {}) {
  return get(`/experiments/${experimentId}/feature-importance`, params)
}
export function fetchShapInfo(experimentId) {
  return get(`/experiments/${experimentId}/shap`)
}
export function fetchLocalExplanations(experimentId, params = {}) {
  return get(`/experiments/${experimentId}/local-explanations`, params)
}

// ── Spatial ───────────────────────────────────────────────────────────
export function fetchSpatialGrid(params = {}) {
  return get('/spatial-grid', params)
}
export function fetchTimeseries(params = {}) {
  return get('/timeseries', params)
}
export function fetchGridCellDetail(params) {
  return get('/grid-cell-detail', params)
}
