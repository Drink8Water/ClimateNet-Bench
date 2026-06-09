<script setup>
import { ref, onMounted, watch, computed } from 'vue'
import { fetchExperiments, fetchFeatureImportance, fetchShapInfo, fetchLocalExplanations } from '../api/climateApi'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'
import BadgeTag from '../components/common/BadgeTag.vue'

use([BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])

const loading = ref(true)
const error = ref(null)
const experiments = ref([])
const selectedExperiment = ref('synthetic_phase3_baseline')
const selectedModel = ref('')
const selectedRegion = ref('')
const importanceData = ref([])
const shapInfo = ref(null)
const localExplanations = ref([])

async function loadExperiments() {
  try {
    experiments.value = await fetchExperiments()
    const ids = [...new Set(experiments.value.map(e => e.experiment_id))]
    if (ids.length && !ids.includes(selectedExperiment.value)) {
      selectedExperiment.value = ids[0]
    }
  } catch (err) {
    error.value = err.message
  }
}

async function loadAttribution() {
  loading.value = true
  try {
    const params = {}
    if (selectedModel.value) params.model = selectedModel.value
    if (selectedRegion.value) params.region = selectedRegion.value
    const [imp, shap, local] = await Promise.all([
      fetchFeatureImportance(selectedExperiment.value, params),
      fetchShapInfo(selectedExperiment.value),
      fetchLocalExplanations(selectedExperiment.value, params),
    ])
    importanceData.value = imp
    shapInfo.value = shap
    localExplanations.value = local
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

const uniqueModels = computed(() => [...new Set(importanceData.value.map(d => d.model).filter(Boolean))])
const topFeature = computed(() => importanceData.value[0]?.feature || '—')

// Global feature importance chart
const importanceChartOption = computed(() => {
  if (!importanceData.value.length) return {}
  // Aggregate importance by feature
  const featureMap = {}
  importanceData.value.forEach(d => {
    const imp = d.importance || d.importance_mean || 0
    if (!featureMap[d.feature]) featureMap[d.feature] = 0
    featureMap[d.feature] += Math.abs(imp)
  })
  const sorted = Object.entries(featureMap)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value', name: 'Importance' },
    yAxis: { type: 'category', data: sorted.map(s => s[0]).reverse(), axisLabel: { fontSize: 9 } },
    series: [{ type: 'bar', data: sorted.map(s => s[1]).reverse(), itemStyle: { color: '#2563eb' } }],
  }
})

// Regional comparison chart
const regionalChartOption = computed(() => {
  const withRegion = importanceData.value.filter(d => d.region)
  if (!withRegion.length) return {}
  const regions = [...new Set(withRegion.map(d => d.region))]
  const features = [...new Set(withRegion.map(d => d.feature))].slice(0, 10)
  const series = regions.map((r, i) => ({
    name: r,
    type: 'bar',
    data: features.map(f => {
      const match = withRegion.find(d => d.region === r && d.feature === f)
      return match ? Math.abs(match.importance || match.importance_mean || 0) : 0
    }),
    itemStyle: { color: i === 0 ? '#f59e0b' : '#0d9488' },
  }))

  return {
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { fontSize: 10 } },
    grid: { left: '3%', right: '4%', bottom: '8%', top: '15%', containLabel: true },
    xAxis: { type: 'category', data: features, axisLabel: { fontSize: 9, rotate: 30 } },
    yAxis: { type: 'value', name: 'Importance' },
    series,
  }
})

onMounted(async () => {
  await loadExperiments()
  await loadAttribution()
})

watch([selectedExperiment, selectedModel, selectedRegion], loadAttribution)
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading feature attribution..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-5">
    <!-- Filter Row -->
    <div class="card p-4 flex flex-wrap items-end gap-3">
      <div class="flex flex-col gap-1">
        <label class="text-[10px] font-semibold text-gray-500 uppercase">Experiment</label>
        <select v-model="selectedExperiment">
          <option v-for="id in [...new Set(experiments.map(e => e.experiment_id))]" :key="id" :value="id">{{ id }}</option>
        </select>
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-[10px] font-semibold text-gray-500 uppercase">Model</label>
        <select v-model="selectedModel">
          <option value="">All Models</option>
          <option v-for="m in uniqueModels" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-[10px] font-semibold text-gray-500 uppercase">Region</label>
        <select v-model="selectedRegion">
          <option value="">All Regions</option>
          <option value="Sahara">Sahara</option>
          <option value="East China">East China</option>
        </select>
      </div>
    </div>

    <!-- Info Cards -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="card p-3"><span class="text-xs text-gray-500">Top Feature</span><div class="text-sm font-bold mt-1">{{ topFeature }}</div></div>
      <div class="card p-3"><span class="text-xs text-gray-500">SHAP Available</span><div class="text-sm font-bold mt-1">{{ shapInfo?.available ? 'Yes' : 'No' }}</div></div>
      <div class="card p-3"><span class="text-xs text-gray-500">Importance Records</span><div class="text-sm font-bold mt-1">{{ importanceData.length }}</div></div>
      <div class="card p-3"><span class="text-xs text-gray-500">Local Explanations</span><div class="text-sm font-bold mt-1">{{ localExplanations.length }}</div></div>
    </div>

    <!-- Charts -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div class="card p-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-3">Global Feature Importance</h4>
        <VChart :option="importanceChartOption" style="height:400px" autoresize />
      </div>
      <div class="card p-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-3">Regional Importance Comparison</h4>
        <VChart v-if="regionalChartOption.series" :option="regionalChartOption" style="height:400px" autoresize />
        <div v-else class="text-center py-16 text-gray-400 text-sm">No regional breakdown available for this selection.</div>
      </div>
    </div>

    <!-- SHAP Info -->
    <div v-if="shapInfo?.available" class="card p-4">
      <h4 class="text-sm font-semibold text-gray-700 mb-2">SHAP Plots</h4>
      <div class="flex flex-wrap gap-2">
        <BadgeTag v-for="p in shapInfo.plot_paths" :key="p" :text="p" color="purple" />
      </div>
    </div>

    <!-- Local Explanations Table -->
    <div v-if="localExplanations.length" class="card p-4 overflow-x-auto">
      <h4 class="text-sm font-semibold text-gray-700 mb-3">Top Local Explanations</h4>
      <table class="data-table">
        <thead>
          <tr>
            <th>Grid ID</th>
            <th>Region</th>
            <th>Feature</th>
            <th>SHAP Score</th>
            <th>Type</th>
            <th>Dominant</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="exp in localExplanations.slice(0, 20)" :key="exp.grid_id + exp.feature">
            <td class="font-mono text-xs">{{ exp.grid_id }}</td>
            <td>{{ exp.region }}</td>
            <td class="font-mono text-xs">{{ exp.feature }}</td>
            <td class="font-mono text-xs">{{ exp.shap_score?.toFixed(6) }}</td>
            <td><BadgeTag :text="exp.importance_type" color="cyan" /></td>
            <td>{{ exp.dominant ? '✓' : '' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
