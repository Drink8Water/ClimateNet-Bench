<script setup>
import Sidebar from './components/layout/Sidebar.vue'
import TopBar from './components/layout/TopBar.vue'
import { useRoute } from 'vue-router'
import { computed } from 'vue'

const route = useRoute()
const pageTitle = computed(() => route.meta?.title || 'ClimateNet')
</script>

<template>
  <div class="min-h-screen bg-[var(--color-bg)]">
    <Sidebar />
    <div class="min-w-0">
      <TopBar :title="pageTitle" />
      <main id="main-content" class="mx-auto w-full max-w-[1440px] px-5 py-6 lg:px-8">
        <router-view />
      </main>
    </div>
  </div>
</template>
