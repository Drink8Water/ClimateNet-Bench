<script setup>
import { useI18n } from '../../i18n'

defineProps({
  title: { type: String, default: 'ClimateNet' },
})

const { locale } = useI18n()
</script>

<template>
  <section class="mx-auto flex w-full max-w-[1440px] items-center justify-between gap-5 px-5 py-5 lg:px-8">
    <div class="min-w-0">
      <p class="text-[11px] font-semibold tracking-[0.12em] text-[var(--color-muted)]">{{ locale === 'zh' ? '实验分析系统' : 'Experiment analysis' }}</p>
      <h1 class="mt-1 text-xl font-semibold text-[var(--color-text)]">{{ title }}</h1>
    </div>
    <div class="flex items-center gap-4">
      <div class="relative hidden md:block">
        <input
          type="text"
          placeholder="Filter experiments"
          class="w-60 pl-8 pr-3 py-1.5 text-sm bg-[var(--color-panel)] border-[var(--color-border)] rounded-lg focus:bg-white"
          disabled
        />
        <span class="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-muted)] text-sm">⌕</span>
      </div>
      <div class="status-chip">
        {{ locale === 'zh' ? 'API 就绪' : 'API ready' }}
      </div>
      <div class="hidden h-8 w-8 rounded-lg bg-[var(--color-accent)] text-white text-xs font-semibold sm:grid place-items-center">
        ML
      </div>
    </div>
  </section>
</template>
