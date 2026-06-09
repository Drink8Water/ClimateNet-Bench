<script setup>
import { ref, onMounted } from 'vue'
import { fetchProjectSummary, fetchDatasetSummary, fetchExperimentSummary } from '../api/climateApi'
import MetricCard from '../components/common/MetricCard.vue'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'
import BadgeTag from '../components/common/BadgeTag.vue'

const project = ref(null)
const dataset = ref(null)
const expSummary = ref(null)
const loading = ref(true)
const error = ref(null)

onMounted(async () => {
  try {
    const [p, d, e] = await Promise.all([
      fetchProjectSummary(),
      fetchDatasetSummary(),
      fetchExperimentSummary(),
    ])
    project.value = p
    dataset.value = d
    expSummary.value = e
  } catch (err) {
    error.value = err.message || 'Failed to load dashboard data'
  } finally {
    loading.value = false
  }
})

const pipelineSteps = [
  { step: 1, label: 'ERA5-Land', desc: 'NetCDF climate reanalysis' },
  { step: 2, label: 'Preprocessing', desc: 'xarray + anomaly calc' },
  { step: 3, label: 'Features', desc: 'Physical predictors' },
  { step: 4, label: 'ML Models', desc: 'LR, RF, XGB, LGBM, TCN' },
  { step: 5, label: 'FastAPI', desc: 'REST API serving results' },
  { step: 6, label: 'Dashboard', desc: 'Vue 3 interactive viz' },
]
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading ClimateNet overview..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-6">
    <!-- Hero -->
    <div class="card p-6">
      <h2 class="text-2xl font-bold text-gray-900">{{ project?.title || 'ClimateNet' }}</h2>
      <p class="text-sm text-gray-500 mt-1">{{ project?.subtitle || '' }}</p>
      <p class="text-sm text-gray-600 mt-3 leading-relaxed max-w-3xl">{{ project?.description || '' }}</p>
    </div>

    <!-- Dataset KPIs -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCard label="Data Source" :value="dataset?.data_source || 'ERA5-Land'" />
      <MetricCard label="Regions" :value="dataset?.regions?.length || 0" unit="regions" />
      <MetricCard label="Target" :value="dataset?.target_variable || 'evaporation_anomaly'" />
      <MetricCard label="Total Records" :value="dataset?.total_records?.toLocaleString() || '0'" unit="rows" />
    </div>

    <!-- Experiment KPIs -->
    <div v-if="expSummary" class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCard label="Experiments" :value="expSummary.total_experiments" />
      <MetricCard label="Best R²" :value="expSummary.best_r2?.toFixed(3)" />
      <MetricCard label="Best RMSE" :value="expSummary.best_rmse?.toFixed(3)" />
      <MetricCard label="Best Model" :value="expSummary.best_model" />
    </div>

    <!-- Models & Strategies -->
    <div v-if="expSummary" class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div class="card p-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-2">Models</h4>
        <div class="flex flex-wrap gap-2">
          <BadgeTag v-for="m in expSummary.models" :key="m" :text="m" :color="m === 'tcn' ? 'purple' : m === 'xgboost' ? 'teal' : 'blue'" />
        </div>
      </div>
      <div class="card p-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-2">Validation Strategies</h4>
        <div class="flex flex-wrap gap-2">
          <BadgeTag v-for="s in expSummary.strategies" :key="s" :text="s" :color="s === 'region_transfer' ? 'amber' : s === 'temporal_holdout' ? 'purple' : 'cyan'" />
        </div>
      </div>
    </div>

    <!-- Pipeline Diagram -->
    <div class="card p-6">
      <h3 class="text-base font-semibold text-gray-900 mb-4">Pipeline</h3>
      <div class="flex flex-wrap items-center gap-2">
        <template v-for="(step, i) in pipelineSteps" :key="step.step">
          <div class="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-50 border border-blue-100">
            <span class="w-5 h-5 rounded-full bg-blue-600 text-white text-xs flex items-center justify-center font-bold">{{ step.step }}</span>
            <div>
              <div class="text-xs font-semibold text-gray-800">{{ step.label }}</div>
              <div class="text-[10px] text-gray-500">{{ step.desc }}</div>
            </div>
          </div>
          <span v-if="i < pipelineSteps.length - 1" class="text-gray-300 font-bold text-lg">→</span>
        </template>
      </div>
    </div>

    <!-- Architecture -->
    <div class="card p-6">
      <h3 class="text-base font-semibold text-gray-900 mb-4">Architecture</h3>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        <div v-for="layer in project?.architecture_layers || []" :key="layer.layer" class="p-3 rounded-lg bg-gray-50 border border-gray-100">
          <div class="text-xs font-semibold text-gray-800">{{ layer.layer }}</div>
          <div class="text-[10px] text-gray-500 mt-1">{{ layer.description }}</div>
          <div class="flex flex-wrap gap-1 mt-2">
            <BadgeTag v-for="tech in (layer.technologies || [])" :key="tech" :text="tech" color="blue" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
