import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Overview',
    component: () => import('../views/Overview.vue'),
    meta: { title: 'Overview' },
  },
  {
    path: '/experiments',
    name: 'ExperimentDashboard',
    component: () => import('../views/ExperimentDashboard.vue'),
    meta: { title: 'Experiment Dashboard' },
  },
  {
    path: '/comparison',
    name: 'ModelComparison',
    component: () => import('../views/ModelComparison.vue'),
    meta: { title: 'Model Comparison' },
  },
  {
    path: '/predictions',
    name: 'PredictionExplorer',
    component: () => import('../views/PredictionExplorer.vue'),
    meta: { title: 'Prediction Explorer' },
  },
  {
    path: '/attribution',
    name: 'FeatureAttribution',
    component: () => import('../views/FeatureAttribution.vue'),
    meta: { title: 'Feature Attribution' },
  },
  {
    path: '/spatial',
    name: 'SpatialViewer',
    component: () => import('../views/SpatialViewer.vue'),
    meta: { title: 'Spatial Viewer' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
