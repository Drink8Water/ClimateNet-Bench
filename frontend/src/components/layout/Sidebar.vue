<script setup>
import { useRoute, useRouter } from 'vue-router'
import { computed } from 'vue'

const route = useRoute()
const router = useRouter()

const navItems = [
  { path: '/', label: 'Overview', icon: '⊡' },
  { path: '/leaderboard', label: 'Leaderboard', icon: '⊟' },
  { path: '/split-difficulty', label: 'Split Difficulty', icon: '⊞' },
  { path: '/forecast', label: 'Forecast Explorer', icon: '◷' },
  { path: '/uncertainty', label: 'Uncertainty', icon: '◎' },
  { path: '/physical', label: 'Physical Audit', icon: '⚖' },
  { path: '/spatial', label: 'Spatial Diagnostics', icon: '◫' },
]

function isActive(path) {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}
function navigate(path) { router.push(path) }
</script>

<template>
  <aside class="w-56 flex-shrink-0 bg-white border-r border-gray-200 flex flex-col h-full">
    <div class="px-4 py-4 border-b border-gray-100">
      <div class="flex items-center gap-2.5">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-teal-500 flex items-center justify-center text-white font-bold text-sm">C</div>
        <div>
          <div class="font-semibold text-sm text-gray-900">ClimateNet-Bench</div>
          <div class="text-[10px] text-gray-400">Evaporation Anomaly Forecast</div>
        </div>
      </div>
    </div>

    <nav class="flex-1 px-3 py-3 space-y-0.5 overflow-y-auto">
      <div
        v-for="item in navItems"
        :key="item.path"
        class="sidebar-link"
        :class="{ active: isActive(item.path) }"
        @click="navigate(item.path)"
      >
        <span class="text-base w-5 text-center">{{ item.icon }}</span>
        <span>{{ item.label }}</span>
      </div>
    </nav>

    <div class="px-4 py-3 border-t border-gray-100 text-xs text-gray-400">
      ClimateNet-Bench v0.3.0
    </div>
  </aside>
</template>
