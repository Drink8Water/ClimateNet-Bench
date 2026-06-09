<script setup>
import { ref, onMounted } from 'vue'
import { fetchCalibration, fetchLeaderboard } from '../api/climateApi'
import * as echarts from 'echarts'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'

const calib = ref([]); const lb = ref([]); const loading = ref(true); const error = ref('')
let chartInstance = null

onMounted(async () => {
  try {
    [calib.value, lb.value] = await Promise.all([fetchCalibration(), fetchLeaderboard({ limit: 500 })])
  } catch (e) { error.value = e.message } finally { loading.value = false }
  setTimeout(renderChart, 200)
})

function renderChart() {
  const el = document.getElementById('coverage-chart')
  if (!el) return
  // Use leaderboard data if calibration is empty
  const data = calib.value.length ? calib.value : lb.value.filter(r => r.coverage_90 != null && r.mean_interval_width != null)
  if (!data.length) return
  if (chartInstance) chartInstance.dispose()
  chartInstance = echarts.init(el)
  chartInstance.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'value', name: 'Mean Interval Width', axisLabel: { fontSize: 11 } },
    yAxis: { type: 'value', name: 'Coverage', min: 0, max: 1, axisLabel: { fontSize: 11 } },
    series: [{
      type: 'scatter', symbolSize: 12,
      data: data.slice(0, 100).map(r => [r.mean_interval_width || 0, r.coverage_90 || 0]),
      itemStyle: { color: '#2563eb', opacity: 0.7 }
    }],
    markLine: { silent: true, data: [{ yAxis: 0.9, lineStyle: { color: '#ef4444', type: 'dashed' }, label: { formatter: 'Target 90%', fontSize: 10 } }] },
    grid: { left: 60, right: 20, top: 20, bottom: 40 }
  })
  window.addEventListener('resize', () => chartInstance?.resize())
}
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading calibration data..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-4">
    <div><h1 class="text-2xl font-bold text-gray-900">Uncertainty Calibration</h1>
      <p class="text-gray-500 text-sm mt-1">Coverage vs interval width — how well are prediction intervals calibrated?</p>
    </div>

    <div class="card p-4" v-if="calib.length || lb.some(r => r.coverage_90 != null)">
      <div id="coverage-chart" style="height:400px"></div>
      <p class="text-xs text-gray-400 mt-2">Red dashed line = target 90% coverage. Points near this line with narrow intervals = best calibration.</p>
    </div>
    <div v-else class="card p-8 text-center text-gray-400">
      No uncertainty calibration data yet. Run the benchmark with conformal prediction enabled.
    </div>

    <div class="card p-4 bg-blue-50 border-blue-200">
      <p class="text-sm text-blue-800"><strong>Split conformal prediction</strong> uses the validation set to calibrate a constant-width prediction interval. Coverage is guaranteed in expectation under exchangeability, but may degrade under OOD splits.</p>
    </div>
  </div>
</template>
