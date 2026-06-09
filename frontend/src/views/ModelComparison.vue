<script setup>
import { ref, onMounted, watch, computed } from 'vue'
import { fetchModelComparison, fetchAblationStudy } from '../api/climateApi'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { BarChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'

use([BarChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])

const loading = ref(true)
const error = ref(null)
const comparisonData = ref([])
const ablationData = ref([])
const selectedMetric = ref('rmse')
const selectedStrategy = ref('')

async function loadData() {
  loading.value = true
  try {
    const [comp, abl] = await Promise.all([
      fetchModelComparison({ metric: selectedMetric.value }),
      fetchAblationStudy(),
    ])
    comparisonData.value = comp
    ablationData.value = abl
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

// Bar chart: model × feature_set
const modelFeatureOption = computed(() => {
  const data = comparisonData.value
  if (!data.length) return {}
  const models = [...new Set(data.map(d => d.model_name))]
  const featSets = [...new Set(data.map(d => d.feature_set))]
  const series = featSets.map(fs => ({
    name: fs,
    type: 'bar',
    data: models.map(m => {
      const match = data.find(d => d.model_name === m && d.feature_set === fs)
      return match ? match.value : null
    }),
  }))

  return {
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    grid: { left: '3%', right: '4%', bottom: '10%', top: '15%', containLabel: true },
    xAxis: { type: 'category', data: models, axisLabel: { fontSize: 10 } },
    yAxis: { type: 'value', name: selectedMetric.value.toUpperCase(), nameTextStyle: { fontSize: 10 } },
    series,
  }
})

// Bar chart: model × validation strategy
const modelValidationOption = computed(() => {
  const data = comparisonData.value
  if (!data.length) return {}
  const models = [...new Set(data.map(d => d.model_name))]
  const strategies = [...new Set(data.map(d => d.validation_strategy))]
  const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666']

  const series = strategies.map((s, i) => ({
    name: s,
    type: 'bar',
    data: models.map(m => {
      const matches = data.filter(d => d.model_name === m && d.validation_strategy === s)
      if (!matches.length) return null
      return matches.reduce((sum, d) => sum + (d.value || 0), 0) / matches.length
    }),
    itemStyle: { color: colors[i % colors.length] },
  }))

  return {
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    grid: { left: '3%', right: '4%', bottom: '10%', top: '15%', containLabel: true },
    xAxis: { type: 'category', data: models, axisLabel: { fontSize: 10 } },
    yAxis: { type: 'value', name: selectedMetric.value.toUpperCase(), nameTextStyle: { fontSize: 10 } },
    series,
  }
})

// Ablation: horizontal bar
function getAblationOption() {
  const data = ablationData.value
  if (!data.length) return {}
  // Aggregate mean R² by feature_set
  const grouped = {}
  data.forEach(d => {
    if (!grouped[d.feature_set]) grouped[d.feature_set] = []
    grouped[d.feature_set].push(d.mean_r2 || 0)
  })
  const categories = []
  const values = []
  Object.entries(grouped).forEach(([k, v]) => {
    categories.push(k)
    values.push(v.reduce((a, b) => a + b, 0) / v.length)
  })

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value', name: 'Mean R²', nameTextStyle: { fontSize: 10 } },
    yAxis: { type: 'category', data: categories, axisLabel: { fontSize: 10 } },
    series: [{ type: 'bar', data: values, itemStyle: { color: '#0d9488' } }],
  }
}

onMounted(loadData)
watch([selectedMetric, selectedStrategy], () => loadData())
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading model comparison data..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-5">
    <!-- Metric Selector -->
    <div class="card p-4 flex flex-wrap items-center gap-4">
      <div class="flex flex-col gap-1">
        <label class="text-[10px] font-semibold text-gray-500 uppercase">Metric</label>
        <select v-model="selectedMetric">
          <option value="r2">R²</option>
          <option value="rmse">RMSE</option>
          <option value="mae">MAE</option>
        </select>
      </div>
    </div>

    <!-- Charts Row -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div class="card p-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-3">Model Performance by Feature Set</h4>
        <VChart :option="modelFeatureOption" style="height:300px" autoresize />
      </div>
      <div class="card p-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-3">Model Performance by Validation Strategy</h4>
        <VChart :option="modelValidationOption" style="height:300px" autoresize />
      </div>
    </div>

    <!-- Ablation Study -->
    <div class="card p-4">
      <h4 class="text-sm font-semibold text-gray-700 mb-3">Ablation Study — Feature Set Impact (Mean R²)</h4>
      <VChart :option="getAblationOption()" style="height:250px" autoresize />
    </div>

    <!-- Interpretation Note -->
    <div class="card p-4 bg-blue-50 border-blue-200">
      <div class="flex items-start gap-2">
        <span class="text-blue-600 text-sm font-bold">ℹ</span>
        <div>
          <h5 class="text-sm font-semibold text-blue-900">Why spatial and temporal validation are harder</h5>
          <p class="text-xs text-blue-700 mt-1 leading-relaxed">
            <strong>Random split</strong> allows the model to see nearby grid points from both regions during training,
            leaking spatial information. <strong>Spatial holdout</strong> reserves entire grid cells for testing,
            forcing the model to generalize to unseen locations. <strong>Temporal holdout</strong> tests on future
            years that the model has never seen. <strong>Region transfer</strong> is the strictest test — training
            on one climate region (e.g., Sahara) and evaluating on another (e.g., East China) reveals whether
            the model has learned universal physical relationships or merely memorized regional patterns.
          </p>
        </div>
      </div>
    </div>
  </div>
</template>
