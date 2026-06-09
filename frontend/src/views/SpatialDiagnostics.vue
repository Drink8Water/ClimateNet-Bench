<script setup>
import { ref, onMounted } from 'vue'
import { fetchTimeseries } from '../api/climateApi'
import * as echarts from 'echarts'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'

const tsData = ref([]); const loading = ref(true); const error = ref('')
let chartInstance = null

onMounted(async () => {
  try { tsData.value = await fetchTimeseries({ region: 'Sahara', limit: 200 }) } catch (e) { error.value = e.message } finally { loading.value = false }
  setTimeout(renderChart, 200)
})

function renderChart() {
  const el = document.getElementById('spatial-ts-chart')
  if (!el || !tsData.value.length) return
  if (chartInstance) chartInstance.dispose()
  chartInstance = echarts.init(el)
  const sorted = tsData.value.slice(0, 120)
  const labels = sorted.map(p => `${p.year || p.target_year || '?'}-${String(p.month || p.target_month || 1).padStart(2, '0')}`)
  chartInstance.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: labels, axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: 'value', name: 'Anomaly' },
    series: [{
      type: 'line', data: sorted.map(p => p.evaporation_anomaly || p.y_true || p.actual || 0),
      lineStyle: { color: '#0d9488' }, symbol: 'none', name: 'Evap Anomaly'
    }],
    grid: { left: 60, right: 20, top: 20, bottom: 60 }
  })
  window.addEventListener('resize', () => chartInstance?.resize())
}
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading spatial data..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-4">
    <div><h1 class="text-2xl font-bold text-gray-900">Spatial Diagnostics</h1>
      <p class="text-gray-500 text-sm mt-1">Grid-cell level time series and spatial patterns</p>
    </div>

    <div v-if="tsData.length" class="card p-4">
      <p class="text-sm font-medium text-gray-700 mb-2">Time Series — Sahara</p>
      <div id="spatial-ts-chart" style="height:350px"></div>
    </div>
    <div v-else class="card p-8 text-center text-gray-400">No timeseries data available. Ensure the backend is serving ERA5-Land or synthetic data.</div>

    <p class="text-xs text-gray-400">{{ tsData.length }} data points</p>
  </div>
</template>
