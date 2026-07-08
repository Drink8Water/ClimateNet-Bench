import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'Overview', component: () => import('../views/Overview.vue'), meta: { title: '实验工作台' } },
  { path: '/leaderboard', name: 'Leaderboard', component: () => import('../views/Leaderboard.vue'), meta: { title: '排行榜' } },
  { path: '/evaluation', name: 'EvaluationPlatform', component: () => import('../views/EvaluationPlatform.vue'), meta: { title: '提交评测' } },
  { path: '/evaluation/:id', name: 'EvaluationDetail', component: () => import('../views/EvaluationDetail.vue'), meta: { title: '评测详情' } },
  { path: '/split-difficulty', name: 'SplitDifficulty', component: () => import('../views/SplitDifficulty.vue'), meta: { title: '划分难度' } },
  { path: '/forecast', name: 'ForecastExplorer', component: () => import('../views/ForecastExplorer.vue'), meta: { title: '预测分析' } },
  { path: '/uncertainty', name: 'Uncertainty', component: () => import('../views/UncertaintyCalibration.vue'), meta: { title: '不确定性校准' } },
  { path: '/physical', name: 'PhysicalAudit', component: () => import('../views/PhysicalAudit.vue'), meta: { title: '物理一致性' } },
  { path: '/spatial', name: 'Spatial', component: () => import('../views/SpatialDiagnostics.vue'), meta: { title: '空间诊断' } },
  // Legacy routes — keep working
  { path: '/experiments', redirect: '/forecast' },
  { path: '/comparison', redirect: '/leaderboard' },
  { path: '/predictions', redirect: '/forecast' },
  { path: '/attribution', redirect: '/physical' },
]

const router = createRouter({ history: createWebHistory(), routes })
export default router
