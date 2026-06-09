<script setup>
import { ref, onMounted } from 'vue'
import { fetchPhysicalSummary, fetchRegionalSensitivity } from '../api/climateApi'
import * as echarts from 'echarts'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'

const summary = ref(null); const sensitivity = ref([])
const loading = ref(true); const error = ref('')
let chartInstance = null

onMounted(async () => {
  try {
    const [s, rs] = await Promise.all([fetchPhysicalSummary(), fetchRegionalSensitivity()])
    summary.value = s; sensitivity.value = rs
  } catch (e) { error.value = e.message } finally { loading.value = false }
  setTimeout(renderChart, 200)
})

function renderChart() {
  const el = document.getElementById('regional-chart')
  if (!el || !sensitivity.value.length) return
  if (chartInstance) chartInstance.dispose()
  chartInstance = echarts.init(el)

  const features = [...new Set(sensitivity.value.map(s => s.feature))]
  const regions = [...new Set(sensitivity.value.map(s => s.region))]
  const series = regions.map(reg => ({
    name: reg, type: 'line',
    data: sensitivity.value.filter(s => s.region === reg && s.feature === features[0]).map(s => [s.feature_value, s.mean_prediction]),
    symbol: 'none', smooth: true
  }))

  // For simplicity show first feature
  chartInstance.setOption({
    title: { text: features[0] || 'Regional Sensitivity', textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'value', name: 'Feature Value' },
    yAxis: { type: 'value', name: 'Mean Prediction' },
    series,
    grid: { left: 60, right: 20, top: 40, bottom: 40 }
  })
  window.addEventListener('resize', () => chartInstance?.resize())
}
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading physical audit..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-4">
    <div><h1 class="text-2xl font-bold text-gray-900">Physical Consistency Audit</h1>
      <p class="text-gray-500 text-sm mt-1">Model behaviour diagnostic — NOT causal discovery</p>
    </div>

    <!-- Score -->
    <div v-if="summary?.consistency_score != null" class="card p-5">
      <div class="flex items-center gap-4">
        <div class="text-4xl font-bold" :class="summary.consistency_score >= 0.5 ? 'text-teal-600' : 'text-amber-600'">{{ (summary.consistency_score * 100).toFixed(0) }}%</div>
        <div><p class="font-semibold text-gray-900">Consistency Score</p><p class="text-sm text-gray-500">{{ summary.n_physically_consistent }} / {{ summary.n_features_audited }} features match physical expectations</p></div>
      </div>
    </div>

    <!-- Feature results -->
    <div v-if="summary?.feature_results" class="card overflow-x-auto">
      <table class="data-table">
        <thead><tr><th>Feature</th><th>Expected</th><th>Spearman ρ</th><th>Direction</th><th>Consistent</th></tr></thead>
        <tbody>
          <tr v-for="f in summary.feature_results" :key="f.feature">
            <td class="font-medium text-sm">{{ f.label }}</td>
            <td><span class="badge" :class="f.expected_sign === 'positive' ? 'badge-teal' : 'badge-amber'">{{ f.expected_sign }}</span></td>
            <td class="font-mono">{{ f.spearman_rho?.toFixed(3) }}</td>
            <td>{{ f.direction }}</td>
            <td>{{ f.physically_consistent ? '✅' : '⚠️' }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="sensitivity.length" class="card p-4">
      <div id="regional-chart" style="height:350px"></div>
    </div>

    <div class="card p-4 bg-amber-50 border-amber-200">
      <p class="text-sm text-amber-800"><strong>⚠ This is not causal discovery.</strong> Feature sensitivity curves show how model predictions respond to perturbations. A model may learn a physically plausible correlation without understanding the underlying physics. Domain expert review is recommended.</p>
    </div>
  </div>
</template>
