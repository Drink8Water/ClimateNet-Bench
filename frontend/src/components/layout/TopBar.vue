<script setup>
defineProps({
  title: { type: String, default: 'ClimateNet' },
})
</script>

<template>
  <header class="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6 flex-shrink-0">
    <h1 class="text-lg font-semibold text-gray-900">{{ title }}</h1>
    <div class="flex items-center gap-4">
      <!-- Search (decorative) -->
      <div class="relative hidden md:block">
        <input
          type="text"
          placeholder="Search experiments..."
          class="w-56 pl-8 pr-3 py-1.5 text-sm bg-gray-50 border-gray-200 rounded-lg focus:bg-white"
          disabled
        />
        <span class="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400 text-sm">⌕</span>
      </div>
      <!-- Notification icon -->
      <div class="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-gray-400 text-sm cursor-default">
        ⏉
      </div>
      <!-- User placeholder -->
      <div class="w-8 h-8 rounded-full bg-gradient-to-br from-teal-400 to-cyan-500 flex items-center justify-center text-white text-xs font-semibold">
        ML
      </div>
    </div>
  </header>
</template>
