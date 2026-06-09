import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'Overview', component: () => import('../views/Overview.vue'), meta: { title: 'Benchmark Overview' } },
  { path: '/leaderboard', name: 'Leaderboard', component: () => import('../views/Leaderboard.vue'), meta: { title: 'Leaderboard' } },
  { path: '/split-difficulty', name: 'SplitDifficulty', component: () => import('../views/SplitDifficulty.vue'), meta: { title: 'Split Difficulty' } },
  { path: '/forecast', name: 'ForecastExplorer', component: () => import('../views/ForecastExplorer.vue'), meta: { title: 'Forecast Explorer' } },
  { path: '/uncertainty', name: 'Uncertainty', component: () => import('../views/UncertaintyCalibration.vue'), meta: { title: 'Uncertainty Calibration' } },
  { path: '/physical', name: 'PhysicalAudit', component: () => import('../views/PhysicalAudit.vue'), meta: { title: 'Physical Consistency' } },
  { path: '/spatial', name: 'Spatial', component: () => import('../views/SpatialDiagnostics.vue'), meta: { title: 'Spatial Diagnostics' } },
  // Legacy routes — keep working
  { path: '/experiments', redirect: '/forecast' },
  { path: '/comparison', redirect: '/leaderboard' },
  { path: '/predictions', redirect: '/forecast' },
  { path: '/attribution', redirect: '/physical' },
]

const router = createRouter({ history: createWebHistory(), routes })
export default router
