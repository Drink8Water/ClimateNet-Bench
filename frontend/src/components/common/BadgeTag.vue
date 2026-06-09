<script setup>
import { computed } from 'vue'

const props = defineProps({
  text: { type: String, required: true },
  color: { type: String, default: 'blue' },
})

const colorMap = {
  blue: 'badge-blue',
  teal: 'badge-teal',
  cyan: 'badge-cyan',
  purple: 'badge-purple',
  amber: 'badge-amber',
}

const badgeClass = computed(() => colorMap[props.color] || colorMap.blue)
</script>

<template>
  <span class="badge" :class="badgeClass">{{ text }}</span>
</template>
