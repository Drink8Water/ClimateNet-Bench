<script setup>
import { ref, onMounted } from 'vue'
import { fetchSplitDifficulty, fetchLeaderboard } from '../api/climateApi'
import * as echarts from 'echarts'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'

const diff = ref([]); const loading = ref(true); const error = ref('')
let chartInstance = null

onMounted(async () => {
  try { diff.value = await fetchSplitDifficulty() } catch (e) { error.value = e.message } finally { loading.value = false }
  setTimeout(renderChart, 200)
})

function renderChart() {
  const el = document.getElementById('split-diff-chart')
  if (!el || !diff.value.length) return
  if (chartInstance) chartInstance.dispose()
  chartInstance = echarts.init(el)
  chartInstance.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: diff.value.map(d => d.split_protocol), axisLabel: { rotate: 15, fontSize: 11 } },
    yAxis: { type: 'value', name: 'Mean RMSE', axisLabel: { fontSize: 11 } },
    series: [{
      type: 'bar', data: diff.value.map(d => d.mean_rmse),
      itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#2563eb' }, { offset: 1, color: '#0d9488' }]) },
      label: { show: true, position: 'top', fontSize: 11, formatter: p => p.value.toFixed(3) }
    }],
    grid: { left: 60, right: 20, top: 20, bottom: 80 }
  })
  window.addEventListener('resize', () => chartInstance?.resize())
}
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading split difficulty..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-4">
    <div><h1 class="text-2xl font-bold text-gray-900">Split Difficulty Analysis</h1>
      <p class="text-gray-500 text-sm mt-1">How much harder are OOD splits compared to in-distribution baselines?</p>
    </div>

    <div class="card p-4">
      <div id="split-diff-chart" style="height:350px"></div>
    </div>

    <div class="card overflow-x-auto">
      <table class="data-table">
        <thead><tr><th>Split Protocol</th><th>Mean RMSE</th><th>Std RMSE</th><th>Min RMSE</th><th>Max RMSE</th><th># Models</th></tr></thead>
        <tbody>
          <tr v-for="d in diff" :key="d.split_protocol">
            <td class="font-medium"><span class="badge badge-teal">{{ d.split_protocol }}</span></td>
            <td class="font-mono">{{ d.mean_rmse?.toFixed(4) }}</td>
            <td class="font-mono text-gray-500">{{ d.std_rmse?.toFixed(4) }}</td>
            <td class="font-mono">{{ d.min_rmse?.toFixed(4) }}</td>
            <td class="font-mono">{{ d.max_rmse?.toFixed(4) }}</td>
            <td>{{ d.n_models }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card p-4 bg-amber-50 border-amber-200">
      <p class="text-sm text-amber-800"><strong>Why random split is optimistic:</strong> Random splitting leaks spatial and temporal information because nearby grid cells and adjacent months are highly correlated. The <code>spatiotemporal</code> protocol (hold out BOTH unseen locations AND future years) is the strictest test of generalization.</p>
    </div>
  </div>
</template>
