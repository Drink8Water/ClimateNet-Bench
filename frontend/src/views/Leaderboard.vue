<script setup>
import { ref, onMounted, watch } from 'vue'
import { fetchLeaderboard, fetchBenchmarkSummary } from '../api/climateApi'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'

const rows = ref([]); const meta = ref(null)
const loading = ref(true); const error = ref('')
const splitFilter = ref(''); const modelFilter = ref('')

onMounted(async () => {
  try {
    const [lb, m] = await Promise.all([fetchLeaderboard({ limit: 500 }), fetchBenchmarkSummary()])
    rows.value = lb; meta.value = m
  } catch (e) { error.value = e.message } finally { loading.value = false }
})

const filtered = () => {
  let r = rows.value
  if (splitFilter.value) r = r.filter(x => x.split_protocol === splitFilter.value)
  if (modelFilter.value) r = r.filter(x => x.model_name === modelFilter.value)
  return r
}

const unique = (key) => [...new Set(rows.value.map(r => r[key]))].sort()

function fmt(v) { return typeof v === 'number' ? v.toFixed(4) : v || '—' }
function bestInSplit(split) {
  const inSplit = rows.value.filter(r => r.split_protocol === split)
  if (!inSplit.length) return null
  return inSplit.reduce((a, b) => (a.rmse || 99) < (b.rmse || 99) ? a : b)
}
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading leaderboard..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-4">
    <div><h1 class="text-2xl font-bold text-gray-900">Benchmark Leaderboard</h1>
      <p class="text-gray-500 text-sm mt-1">Ranked by RMSE within each split protocol</p>
    </div>

    <!-- Filters -->
    <div class="flex gap-3 flex-wrap">
      <select v-model="splitFilter"><option value="">All Splits</option><option v-for="s in unique('split_protocol')" :key="s" :value="s">{{ s }}</option></select>
      <select v-model="modelFilter"><option value="">All Models</option><option v-for="m in unique('model_name')" :key="m" :value="m">{{ m }}</option></select>
    </div>

    <!-- Best per split -->
    <div class="grid grid-cols-2 lg:grid-cols-3 gap-3">
      <div v-for="s in unique('split_protocol')" :key="s" class="card p-3">
        <span class="badge badge-blue text-[10px]">{{ s }}</span>
        <span class="text-sm ml-2 font-medium">{{ bestInSplit(s)?.model_name }}</span>
        <span class="text-xs text-gray-400 ml-1">RMSE {{ bestInSplit(s)?.rmse?.toFixed(3) }}</span>
      </div>
    </div>

    <!-- Table -->
    <div class="card overflow-x-auto">
      <table class="data-table">
        <thead>
          <tr><th>Rank</th><th>Model</th><th>Split</th><th>Feature Set</th><th>RMSE ↓</th><th>MAE</th><th>R²</th><th>Skill vs Pers.</th></tr>
        </thead>
        <tbody>
          <tr v-for="r in filtered()" :key="r.experiment_id" :class="{ 'bg-blue-50': r.rank === 1 }">
            <td class="font-bold">{{ r.rank }}</td>
            <td class="font-medium">{{ r.model_name }}</td>
            <td><span class="badge badge-teal">{{ r.split_protocol }}</span></td>
            <td>{{ r.feature_set }}</td>
            <td class="font-mono">{{ fmt(r.rmse) }}</td>
            <td class="font-mono text-gray-500">{{ fmt(r.mae) }}</td>
            <td class="font-mono" :class="r.r2 > 0 ? 'text-teal-600' : 'text-gray-400'">{{ fmt(r.r2) }}</td>
            <td class="font-mono" :class="(r.skill_vs_persistence || 0) > 0 ? 'text-teal-600' : 'text-red-400'">{{ r.skill_vs_persistence != null ? (r.skill_vs_persistence * 100).toFixed(1) + '%' : '—' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="text-xs text-gray-400">{{ filtered().length }} of {{ rows.length }} rows</p>
  </div>
</template>
