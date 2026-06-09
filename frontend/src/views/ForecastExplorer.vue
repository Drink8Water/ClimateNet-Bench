<script setup>
import { ref, onMounted } from 'vue'
import { fetchExperiments, fetchPredictions } from '../api/climateApi'
import * as echarts from 'echarts'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'

const experiments = ref([]); const selectedExp = ref('')
const preds = ref([]); const loading = ref(true); const predLoading = ref(false); const error = ref('')
let scatterChart = null; let timeChart = null

onMounted(async () => {
  try { experiments.value = await fetchExperiments({ limit: 200 }) } catch (e) { error.value = e.message } finally { loading.value = false }
})

async function loadPredictions() {
  if (!selectedExp.value) return
  predLoading.value = true
  try { preds.value = await fetchPredictions(selectedExp.value, { limit: 1000 }) } catch (e) { error.value = e.message } finally { predLoading.value = false }
  setTimeout(renderCharts, 200)
}

function renderCharts() {
  // Scatter: predicted vs actual
  const el1 = document.getElementById('scatter-chart')
  if (el1 && preds.value.length) {
    if (scatterChart) scatterChart.dispose()
    scatterChart = echarts.init(el1)
    scatterChart.setOption({
      tooltip: { trigger: 'item' },
      xAxis: { type: 'value', name: 'Actual', axisLabel: { fontSize: 11 } },
      yAxis: { type: 'value', name: 'Predicted', axisLabel: { fontSize: 11 } },
      series: [{
        type: 'scatter', data: preds.value.slice(0, 500).map(p => [p.actual || p.y_true, p.prediction || p.y_pred]),
        symbolSize: 4, itemStyle: { color: '#2563eb', opacity: 0.5 }
      }],
      grid: { left: 60, right: 20, top: 20, bottom: 40 }
    })
  }
  // Time series
  const el2 = document.getElementById('time-chart')
  if (el2 && preds.value.length) {
    if (timeChart) timeChart.dispose()
    timeChart = echarts.init(el2)
    const sorted = [...preds.value].sort((a, b) => (a.year || a.target_year) - (b.year || b.target_year) || (a.month || a.target_month) - (b.month || b.target_month))
    const labels = sorted.slice(0, 60).map(p => `${p.year || p.target_year}-${String(p.month || p.target_month).padStart(2, '0')}`)
    const actual = sorted.slice(0, 60).map(p => p.actual || p.y_true)
    const pred = sorted.slice(0, 60).map(p => p.prediction || p.y_pred)
    timeChart.setOption({
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: labels, axisLabel: { rotate: 45, fontSize: 10 } },
      yAxis: { type: 'value', name: 'Evap Anomaly' },
      series: [
        { name: 'Actual', type: 'line', data: actual, lineStyle: { color: '#0d9488' }, itemStyle: { color: '#0d9488' }, symbol: 'none' },
        { name: 'Predicted', type: 'line', data: pred, lineStyle: { color: '#2563eb' }, itemStyle: { color: '#2563eb' }, symbol: 'none' },
      ],
      grid: { left: 60, right: 20, top: 30, bottom: 60 }
    })
  }
  window.addEventListener('resize', () => { scatterChart?.resize(); timeChart?.resize() })
}
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading experiments..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-4">
    <div><h1 class="text-2xl font-bold text-gray-900">Forecast Explorer</h1>
      <p class="text-gray-500 text-sm mt-1">Explore predictions from benchmark experiments</p>
    </div>

    <div class="flex gap-3 items-center">
      <select v-model="selectedExp" @change="loadPredictions" class="min-w-[400px]">
        <option value="">— Select an experiment —</option>
        <option v-for="e in experiments" :key="e.experiment_id" :value="e.experiment_id">{{ e.model_name }} / {{ e.validation_strategy || e.split_protocol }} / {{ e.feature_set }}</option>
      </select>
    </div>

    <div v-if="predLoading"><LoadingState message="Loading predictions..." /></div>
    <div v-else-if="preds.length" class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div class="card p-4"><p class="text-sm font-medium text-gray-700 mb-2">Predicted vs Actual</p><div id="scatter-chart" style="height:300px"></div></div>
      <div class="card p-4"><p class="text-sm font-medium text-gray-700 mb-2">Time Series (first 60 samples)</p><div id="time-chart" style="height:300px"></div></div>
    </div>
    <div v-else-if="selectedExp" class="card p-8 text-center text-gray-400">No predictions available for this experiment</div>
    <p class="text-xs text-gray-400" v-if="preds.length">{{ preds.length }} predictions loaded</p>
  </div>
</template>
