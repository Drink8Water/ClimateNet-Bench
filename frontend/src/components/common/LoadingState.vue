<script setup>
defineProps({
  message: { type: String, default: 'Loading...' },
})
</script>

<template>
  <div class="space-y-4 py-8">
    <div class="h-24 animate-pulse rounded-lg bg-[var(--color-panel)]"></div>
    <div class="grid grid-cols-1 gap-3 md:grid-cols-3">
      <div class="h-20 animate-pulse rounded-lg bg-[var(--color-panel)]"></div>
      <div class="h-20 animate-pulse rounded-lg bg-[var(--color-panel)]"></div>
      <div class="h-20 animate-pulse rounded-lg bg-[var(--color-panel)]"></div>
    </div>
    <span class="block text-sm text-[var(--color-muted)]">{{ message }}</span>
  </div>
</template>
