<script setup>
import { ref, onMounted } from 'vue'
import { fetchBenchmarkSummary, fetchBenchmarkTask, fetchBenchmarkRegions, fetchBenchmarkSplits } from '../api/climateApi'
import MetricCard from '../components/common/MetricCard.vue'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'

const summary = ref(null); const task = ref(null)
const regions = ref([]); const splits = ref([])
const loading = ref(true); const error = ref('')

onMounted(async () => {
  try {
    const [s, t, r, sp] = await Promise.all([
      fetchBenchmarkSummary(), fetchBenchmarkTask(),
      fetchBenchmarkRegions(), fetchBenchmarkSplits()
    ])
    summary.value = s; task.value = t; regions.value = r; splits.value = sp
  } catch (e) { error.value = e.message } finally { loading.value = false }
})
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading benchmark overview..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-6">
    <div>
      <h1 class="text-2xl font-bold text-gray-900">EvapAnomaly-Forecast-v1</h1>
      <p class="text-gray-500 mt-1">A reproducible spatio-temporal ML benchmark for next-month land evaporation anomaly forecasting</p>
    </div>

    <!-- Central Question -->
    <div class="card p-5 bg-gradient-to-r from-blue-50 to-teal-50 border-blue-200">
      <p class="text-sm font-semibold text-blue-900">Central Research Question</p>
      <p class="text-lg text-blue-800 mt-1">Do ML models trained on climate data truly generalize across unseen grid cells, future years, and different climate regions?</p>
    </div>

    <!-- KPIs -->
    <div class="grid grid-cols-2 lg:grid-cols-5 gap-4">
      <MetricCard label="Experiments" :value="summary.total_experiments" />
      <MetricCard label="Models" :value="summary.n_models" />
      <MetricCard label="Split Protocols" :value="summary.n_split_protocols" />
      <MetricCard label="Best RMSE" :value="summary.best_rmse?.toFixed(3)" unit="evap anomaly" />
      <MetricCard label="Best Model" :value="summary.best_model" />
    </div>

    <!-- Task Definition -->
    <div class="card p-5">
      <h2 class="text-lg font-semibold text-gray-900 mb-3">Task Definition</h2>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div><span class="text-gray-400">Input</span><p class="font-medium">{{ task.input_window }}</p></div>
        <div><span class="text-gray-400">Target</span><p class="font-medium">{{ task.target }}</p></div>
        <div><span class="text-gray-400">Temporal Unit</span><p class="font-medium">{{ task.temporal_unit }}</p></div>
        <div><span class="text-gray-400">Forecast Horizon</span><p class="font-medium">{{ task.forecast_horizon }}</p></div>
      </div>
    </div>

    <!-- Regions + Splits -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="card p-5">
        <h2 class="text-lg font-semibold text-gray-900 mb-3">Benchmark Regions ({{ regions.length }})</h2>
        <table class="data-table">
          <thead><tr><th>Region</th><th>Climate</th><th>Bounds</th></tr></thead>
          <tbody>
            <tr v-for="r in regions" :key="r.name">
              <td class="font-medium">{{ r.name }}</td>
              <td><span class="badge badge-teal">{{ r.climate_type }}</span></td>
              <td class="text-xs text-gray-500">lat {{ r.lat_min }}-{{ r.lat_max }}, lon {{ r.lon_min }}-{{ r.lon_max }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="card p-5">
        <h2 class="text-lg font-semibold text-gray-900 mb-3">Split Protocols ({{ splits.length }})</h2>
        <ul class="space-y-2 text-sm">
          <li v-for="s in splits" :key="s.split_id" class="flex items-start gap-2">
            <span class="badge badge-blue mt-0.5">{{ s.protocol }}</span>
            <span class="text-gray-600">{{ s.note?.substring(0, 80) }}...</span>
          </li>
        </ul>
      </div>
    </div>

    <!-- Pipeline -->
    <div class="card p-5">
      <h2 class="text-lg font-semibold text-gray-900 mb-3">Benchmark Pipeline</h2>
      <div class="flex flex-wrap gap-2 text-xs items-center">
        <span class="badge badge-purple">1. ERA5-Land Data</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">2. Feature Engineering</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">3. Forecasting Dataset</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">4. Split Protocols</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">5. Model Training</span><span class="text-gray-300">→</span>
        <span class="badge badge-purple">6. Evaluation</span><span class="text-gray-300">→</span>
        <span class="badge badge-cyan">7. Leaderboard</span>
      </div>
    </div>
  </div>
</template>
