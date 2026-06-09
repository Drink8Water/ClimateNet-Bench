<script setup>
defineProps({
  message: { type: String, default: 'Failed to load data from the server.' },
})
</script>

<template>
  <div class="card p-6 bg-red-50 border-red-200 flex items-start gap-3">
    <span class="text-red-500 text-lg">⚠</span>
    <div>
      <div class="text-sm font-medium text-red-800">Error</div>
      <p class="text-sm text-red-600 mt-1">{{ message }}</p>
      <p class="text-xs text-red-400 mt-2">Make sure the backend is running: uvicorn backend.main:app --port 8000</p>
    </div>
  </div>
</template>
