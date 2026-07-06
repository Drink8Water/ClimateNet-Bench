<script setup>
defineProps({
  message: { type: String, default: 'Failed to load data from the server.' },
})
</script>

<template>
  <div class="card flex items-start gap-3 border-[#e4c8c2] bg-[#fff7f4] p-5">
    <span class="text-[var(--color-danger)] text-lg">!</span>
    <div>
      <div class="text-sm font-semibold text-[var(--color-danger)]">连接或数据加载失败</div>
      <p class="text-sm text-[#8c4c44] mt-1">{{ message }}</p>
      <p class="text-xs text-[#9b6a62] mt-2">请确认后端正在运行：uvicorn backend.main:app --port 8000</p>
    </div>
  </div>
</template>
