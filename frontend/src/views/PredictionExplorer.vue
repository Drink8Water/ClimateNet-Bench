<script setup>
import { ref, onMounted, watch, computed } from 'vue'
import { fetchExperiments, fetchPredictions, fetchResiduals, fetchPredictionSummary } from '../api/climateApi'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { BarChart, ScatterChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'
import MetricCard from '../components/common/MetricCard.vue'

use([BarChart, ScatterChart, LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])

const loading = ref(true)
const error = ref(null)
const experiments = ref([])
const selectedExperiment = ref('synthetic_phase3_baseline')
const selectedRegion = ref('')
const predictions = ref([])
const residuals = ref([])
const summary = ref(null)

async function loadExperiments() {
  try {
    experiments.value = await fetchExperiments()
    // Get unique experiment IDs
    const ids = [...new Set(experiments.value.map(e => e.experiment_id))]
    if (ids.length && !ids.includes(selectedExperiment.value)) {
      selectedExperiment.value = ids[0]
    }
  } catch (err) {
    error.value = err.message
  }
}

async function loadPredictions() {
  loading.value = true
  try {
    const params = { limit: 1000 }
    if (selectedRegion.value) params.region = selectedRegion.value
    const [preds, resids, summ] = await Promise.all([
      fetchPredictions(selectedExperiment.value, params),
      fetchResiduals(selectedExperiment.value, params),
      fetchPredictionSummary(selectedExperiment.value, selectedRegion.value ? { region: selectedRegion.value } : {}),
    ])
    predictions.value = preds
    residuals.value = resids
    summary.value = summ
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

// Scatter chart: actual vs prediction
const scatterOption = computed(() => {
  if (!predictions.value.length) return {}
  const sahara = predictions.value.filter(p => p.region === 'Sahara')
  const eastChina = predictions.value.filter(p => p.region === 'East China')

  const series = []
  if (sahara.length) series.push({ name: 'Sahara', type: 'scatter', data: sahara.map(p => [p.actual, p.prediction]), symbolSize: 5 })
  if (eastChina.length) series.push({ name: 'East China', type: 'scatter', data: eastChina.map(p => [p.actual, p.prediction]), symbolSize: 5 })

  // Diagonal line
  const allActuals = predictions.value.map(p => p.actual).filter(v => v != null)
  const minVal = Math.min(...allActuals)
  const maxVal = Math.max(...allActuals)
  series.push({
    name: 'Perfect',
    type: 'line',
    data: [[minVal, minVal], [maxVal, maxVal]],
    lineStyle: { color: '#94a3b8', type: 'dashed', width: 1 },
    itemStyle: { color: '#94a3b8' },
    symbol: 'none',
    silent: true,
  })

  return {
    tooltip: { trigger: 'item', formatter: p => `Actual: ${p.value[0]?.toFixed(3)}<br/>Predicted: ${p.value[1]?.toFixed(3)}` },
    legend: { top: 0, textStyle: { fontSize: 10 } },
    grid: { left: '3%', right: '4%', bottom: '8%', top: '12%', containLabel: true },
    xAxis: { type: 'value', name: 'Actual', nameTextStyle: { fontSize: 10 } },
    yAxis: { type: 'value', name: 'Predicted', nameTextStyle: { fontSize: 10 } },
    series,
  }
})

// Histogram: residuals
const histogramOption = computed(() => {
  if (!residuals.value.length) return {}
  const values = residuals.value.map(r => r.residual).filter(v => v != null)
  // Simple binning
  const bins = 30
  const min = Math.min(...values)
  const max = Math.max(...values)
  const binWidth = (max - min) / bins
  const counts = Array(bins).fill(0)
  const labels = []
  values.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1)
    counts[idx]++
  })
  for (let i = 0; i < bins; i++) {
    labels.push((min + i * binWidth).toFixed(2))
  }

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '10%', top: '5%', containLabel: true },
    xAxis: { type: 'category', data: labels, axisLabel: { fontSize: 8, rotate: 45 } },
    yAxis: { type: 'value', name: 'Count' },
    series: [{ type: 'bar', data: counts, itemStyle: { color: '#6366f1' } }],
  }
})

// Time series: mean residual over time
const temporalOption = computed(() => {
  if (!residuals.value.length) return {}
  const grouped = {}
  residuals.value.forEach(r => {
    const key = `${r.year}-${String(r.month).padStart(2, '0')}`
    if (!grouped[key]) grouped[key] = { sum: 0, count: 0 }
    if (r.residual != null) {
      grouped[key].sum += r.residual
      grouped[key].count++
    }
  })
  const keys = Object.keys(grouped).sort()
  const means = keys.map(k => grouped[k].sum / grouped[k].count)

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '10%', top: '5%', containLabel: true },
    xAxis: { type: 'category', data: keys, axisLabel: { fontSize: 8, rotate: 45, interval: 11 } },
    yAxis: { type: 'value', name: 'Mean Residual' },
    series: [{ type: 'line', data: means, smooth: true, itemStyle: { color: '#0d9488' } }],
  }
})

onMounted(async () => {
  await loadExperiments()
  await loadPredictions()
})

watch([selectedExperiment, selectedRegion], loadPredictions)
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading predictions..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-5">
    <!-- Controls -->
    <div class="card p-4 flex flex-wrap items-end gap-3">
      <div class="flex flex-col gap-1">
        <label class="text-[10px] font-semibold text-gray-500 uppercase">Experiment</label>
        <select v-model="selectedExperiment">
          <option v-for="id in [...new Set(experiments.map(e => e.experiment_id))]" :key="id" :value="id">{{ id }}</option>
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

    <!-- Summary Cards -->
    <div v-if="summary" class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCard label="Mean Residual" :value="summary.mean_residual?.toFixed(4)" />
      <MetricCard label="Residual Std" :value="summary.residual_std?.toFixed(4)" />
      <MetricCard label="Max Abs Error" :value="summary.max_absolute_error?.toFixed(4)" />
      <MetricCard label="Count" :value="summary.prediction_count?.toLocaleString()" />
    </div>

    <!-- Charts Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div class="card p-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-3">Prediction vs Actual</h4>
        <VChart :option="scatterOption" style="height:350px" autoresize />
      </div>
      <div class="card p-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-3">Residual Distribution</h4>
        <VChart :option="histogramOption" style="height:350px" autoresize />
      </div>
    </div>
    <div class="card p-4">
      <h4 class="text-sm font-semibold text-gray-700 mb-3">Mean Residual Over Time</h4>
      <VChart :option="temporalOption" style="height:250px" autoresize />
    </div>
  </div>
</template>
