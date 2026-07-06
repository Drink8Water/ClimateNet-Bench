<script setup>
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from '../../i18n'

const route = useRoute()
const router = useRouter()
const { locale, setLocale } = useI18n()

const navItems = [
  { path: '/', zh: '概览', en: 'Overview' },
  { path: '/leaderboard', zh: '排行榜', en: 'Leaderboard' },
  { path: '/evaluation', zh: '提交评测', en: 'Evaluation' },
  { path: '/forecast', zh: '预测分析', en: 'Forecasts' },
  { path: '/spatial', zh: '空间诊断', en: 'Spatial' },
  { path: '/uncertainty', zh: '不确定性', en: 'Uncertainty' },
  { path: '/physical', zh: '物理审计', en: 'Physical audit' },
  { path: '/split-difficulty', zh: '划分难度', en: 'Split difficulty' },
]

function isActive(path) {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}
function navigate(path) { router.push(path) }
</script>

<template>
  <header class="sticky top-0 z-30 border-b border-[var(--color-border)] bg-[rgba(245,247,244,0.92)] backdrop-blur">
    <a href="#main-content" class="skip-link">跳到主要内容</a>
    <div class="mx-auto flex min-h-16 max-w-[1440px] flex-wrap items-center gap-3 px-5 py-3 lg:flex-nowrap lg:gap-5 lg:px-8">
      <div class="flex items-center gap-2.5">
        <div class="brand-mark">C</div>
        <div>
          <div class="font-semibold text-sm text-[var(--color-text)]">ClimateNet-Bench</div>
          <div class="text-[10px] text-[var(--color-muted)]">{{ locale === 'zh' ? '气候基准评测工作台' : 'climate evaluation workbench' }}</div>
        </div>
      </div>

      <nav class="order-3 flex w-full items-center gap-1 overflow-x-auto pb-1 lg:order-none lg:w-auto lg:flex-1 lg:pb-0">
        <button
        v-for="item in navItems"
        :key="item.path"
        class="sidebar-link"
        :class="{ active: isActive(item.path) }"
        @click="navigate(item.path)"
      >
          <span>{{ locale === 'zh' ? item.zh : item.en }}</span>
        </button>
      </nav>

      <div class="ml-auto flex items-center gap-2">
        <button class="locale-button" :class="{ active: locale === 'zh' }" @click="setLocale('zh')">中文</button>
        <button class="locale-button" :class="{ active: locale === 'en' }" @click="setLocale('en')">EN</button>
      </div>
    </div>
  </header>
</template>
