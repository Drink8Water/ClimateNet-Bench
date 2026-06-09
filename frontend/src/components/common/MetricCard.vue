<script setup>
defineProps({
  label: { type: String, required: true },
  value: { type: [String, Number], default: '—' },
  unit: { type: String, default: '' },
  accent: { type: String, default: 'blue' },
})
</script>

<template>
  <div class="card p-4 flex flex-col gap-1">
    <span class="text-xs font-medium text-gray-500 uppercase tracking-wide">{{ label }}</span>
    <div class="flex items-baseline gap-1">
      <span class="text-2xl font-bold text-gray-900">{{ value }}</span>
      <span v-if="unit" class="text-sm text-gray-400">{{ unit }}</span>
    </div>
  </div>
</template>
