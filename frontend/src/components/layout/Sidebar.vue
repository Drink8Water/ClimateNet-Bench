<script setup>
import { useRoute, useRouter } from 'vue-router'
import { computed } from 'vue'

const route = useRoute()
const router = useRouter()

const navItems = [
  { path: '/', label: 'Overview', icon: '◉' },
  { path: '/experiments', label: 'Experiments', icon: '⧉' },
  { path: '/comparison', label: 'Model Comparison', icon: '▦' },
  { path: '/predictions', label: 'Predictions', icon: '◷' },
  { path: '/attribution', label: 'Feature Attribution', icon: '◫' },
  { path: '/spatial', label: 'Spatial Viewer', icon: '⊞' },
]

function isActive(path) {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}

function navigate(path) {
  router.push(path)
}
</script>

<template>
  <aside class="w-60 flex-shrink-0 bg-white border-r border-gray-200 flex flex-col h-full">
    <!-- Logo -->
    <div class="px-5 py-5 border-b border-gray-100">
      <div class="flex items-center gap-2.5">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-teal-500 flex items-center justify-center text-white font-bold text-sm">C</div>
        <div>
          <div class="font-semibold text-sm text-gray-900">ClimateNet</div>
          <div class="text-[10px] text-gray-400">ML Experiment Manager</div>
        </div>
      </div>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
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

    <!-- Footer -->
    <div class="px-5 py-3 border-t border-gray-100">
      <span class="badge badge-blue">v0.2.0</span>
    </div>
  </aside>
</template>
