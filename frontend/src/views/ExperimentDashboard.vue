<script setup>
import { ref, onMounted, computed } from 'vue'
import { fetchExperiments, fetchExperimentSummary, fetchExperimentDetail } from '../api/climateApi'
import MetricCard from '../components/common/MetricCard.vue'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'
import BadgeTag from '../components/common/BadgeTag.vue'

const experiments = ref([])
const expSummary = ref(null)
const selectedDetail = ref(null)
const loading = ref(true)
const error = ref(null)

// Filters
const filterModel = ref('')
const filterStrategy = ref('')
const filterFeatureSet = ref('')
const filterRegion = ref('')

const uniqueModels = computed(() => [...new Set(experiments.value.map(e => e.model_name))].sort())
const uniqueStrategies = computed(() => [...new Set(experiments.value.map(e => e.validation_strategy))].sort())
const uniqueFeatureSets = computed(() => [...new Set(experiments.value.map(e => e.feature_set))].sort())
const uniqueRegions = computed(() => {
  const regions = new Set()
  experiments.value.forEach(e => {
    if (e.train_region) e.train_region.split(', ').forEach(r => regions.add(r))
    if (e.test_region) e.test_region.split(', ').forEach(r => regions.add(r))
  })
  return [...regions].sort()
})

const filteredExperiments = computed(() => {
  return experiments.value.filter(e => {
    if (filterModel.value && e.model_name !== filterModel.value) return false
    if (filterStrategy.value && e.validation_strategy !== filterStrategy.value) return false
    if (filterFeatureSet.value && e.feature_set !== filterFeatureSet.value) return false
    if (filterRegion.value) {
      const inTrain = (e.train_region || '').includes(filterRegion.value)
      const inTest = (e.test_region || '').includes(filterRegion.value)
      if (!inTrain && !inTest) return false
    }
    return true
  })
})

function resetFilters() {
  filterModel.value = ''
  filterStrategy.value = ''
  filterFeatureSet.value = ''
  filterRegion.value = ''
}

async function loadData() {
  loading.value = true
  try {
    const [exps, summary] = await Promise.all([
      fetchExperiments(),
      fetchExperimentSummary(),
    ])
    experiments.value = exps
    expSummary.value = summary
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

async function showDetail(exp) {
  try {
    selectedDetail.value = await fetchExperimentDetail(exp.experiment_id)
  } catch (err) {
    selectedDetail.value = { error: err.message }
  }
}

function closeDetail() {
  selectedDetail.value = null
}

function fmtR2(val) {
  if (val == null) return '—'
  return Number(val).toFixed(4)
}

function fmtRMSE(val) {
  if (val == null) return '—'
  return Number(val).toFixed(3)
}

function badgeColor(strategy) {
  if (strategy === 'region_transfer') return 'amber'
  if (strategy === 'temporal_holdout') return 'purple'
  if (strategy === 'spatial_holdout') return 'teal'
  return 'blue'
}

onMounted(loadData)
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading experiments..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-5">
    <!-- KPI Cards -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCard label="Experiments" :value="expSummary?.total_experiments || 0" />
      <MetricCard label="Best R²" :value="expSummary?.best_r2?.toFixed(3) || '—'" />
      <MetricCard label="Best RMSE" :value="expSummary?.best_rmse?.toFixed(3) || '—'" />
      <MetricCard label="Regions" :value="expSummary?.regions?.length || 0" />
    </div>

    <!-- Filter Row -->
    <div class="card p-4">
      <div class="flex flex-wrap items-end gap-3">
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-semibold text-gray-500 uppercase">Model</label>
          <select v-model="filterModel">
            <option value="">All Models</option>
            <option v-for="m in uniqueModels" :key="m" :value="m">{{ m }}</option>
          </select>
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-semibold text-gray-500 uppercase">Strategy</label>
          <select v-model="filterStrategy">
            <option value="">All Strategies</option>
            <option v-for="s in uniqueStrategies" :key="s" :value="s">{{ s }}</option>
          </select>
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-semibold text-gray-500 uppercase">Feature Set</label>
          <select v-model="filterFeatureSet">
            <option value="">All Feature Sets</option>
            <option v-for="f in uniqueFeatureSets" :key="f" :value="f">{{ f }}</option>
          </select>
        </div>
        <div class="flex flex-col gap-1">
          <label class="text-[10px] font-semibold text-gray-500 uppercase">Region</label>
          <select v-model="filterRegion">
            <option value="">All Regions</option>
            <option v-for="r in uniqueRegions" :key="r" :value="r">{{ r }}</option>
          </select>
        </div>
        <button @click="resetFilters" class="px-3 py-1.5 text-xs font-medium text-gray-500 hover:text-gray-700 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors">
          Reset Filters
        </button>
      </div>
    </div>

    <!-- Experiment Table -->
    <div class="card overflow-x-auto">
      <table class="data-table">
        <thead>
          <tr>
            <th>Experiment ID</th>
            <th>Model</th>
            <th>Validation</th>
            <th>Feature Set</th>
            <th>Train Region</th>
            <th>Test Region</th>
            <th>R²</th>
            <th>RMSE</th>
            <th>MAE</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="exp in filteredExperiments"
            :key="exp.experiment_id + exp.model_name + exp.validation_strategy + exp.feature_set"
            @click="showDetail(exp)"
            class="cursor-pointer"
          >
            <td class="font-mono text-xs text-blue-600">{{ exp.experiment_id }}</td>
            <td><BadgeTag :text="exp.model_name" :color="exp.model_name === 'tcn' ? 'purple' : exp.model_name === 'xgboost' ? 'teal' : 'blue'" /></td>
            <td><BadgeTag :text="exp.validation_strategy" :color="badgeColor(exp.validation_strategy)" /></td>
            <td><BadgeTag :text="exp.feature_set" color="cyan" /></td>
            <td class="text-xs text-gray-600 max-w-[120px] truncate">{{ exp.train_region }}</td>
            <td class="text-xs text-gray-600 max-w-[120px] truncate">{{ exp.test_region }}</td>
            <td class="font-mono text-xs">{{ fmtR2(exp.r2) }}</td>
            <td class="font-mono text-xs">{{ fmtRMSE(exp.rmse) }}</td>
            <td class="font-mono text-xs">{{ exp.mae ? Number(exp.mae).toFixed(3) : '—' }}</td>
          </tr>
        </tbody>
      </table>
      <div v-if="filteredExperiments.length === 0" class="text-center py-8 text-gray-400 text-sm">
        No experiments match the selected filters.
      </div>
    </div>

    <!-- Detail Panel -->
    <div v-if="selectedDetail" class="card p-5">
      <div class="flex items-center justify-between mb-3">
        <h4 class="text-sm font-semibold text-gray-800">Experiment Detail: {{ selectedDetail.experiment_id }}</h4>
        <button @click="closeDetail" class="text-gray-400 hover:text-gray-600 text-lg leading-none">&times;</button>
      </div>
      <div v-if="selectedDetail.error" class="text-sm text-red-500">{{ selectedDetail.error }}</div>
      <div v-else class="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
        <div><span class="text-gray-500">Metrics Count:</span> <strong>{{ selectedDetail.metrics_count }}</strong></div>
        <div><span class="text-gray-500">Predictions:</span> <strong>{{ selectedDetail.prediction_count?.toLocaleString() }}</strong></div>
        <div><span class="text-gray-500">Features:</span> <strong>{{ selectedDetail.feature_count?.toLocaleString() }}</strong></div>
      </div>
    </div>
  </div>
</template>
