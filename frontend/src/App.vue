<script setup>
import Sidebar from './components/layout/Sidebar.vue'
import TopBar from './components/layout/TopBar.vue'
import { useRoute } from 'vue-router'
import { computed } from 'vue'

const route = useRoute()
const pageTitle = computed(() => route.meta?.title || 'ClimateNet')
</script>

<template>
  <div class="flex h-screen overflow-hidden">
    <Sidebar />
    <div class="flex-1 flex flex-col min-w-0">
      <TopBar :title="pageTitle" />
      <main class="flex-1 overflow-y-auto p-6">
        <router-view />
      </main>
    </div>
  </div>
</template>
