<script setup>
defineProps({
  message: { type: String, default: 'Loading...' },
})
</script>

<template>
  <div class="flex flex-col items-center justify-center py-20 text-gray-400">
    <div class="w-10 h-10 border-3 border-gray-200 border-t-blue-500 rounded-full animate-spin mb-4"></div>
    <span class="text-sm">{{ message }}</span>
  </div>
</template>
