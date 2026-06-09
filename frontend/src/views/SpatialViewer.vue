<script setup>
import { ref, onMounted, watch, computed } from 'vue'
import { fetchSpatialGrid, fetchTimeseries, fetchGridCellDetail } from '../api/climateApi'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { ScatterChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, VisualMapComponent, GeoComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import LoadingState from '../components/common/LoadingState.vue'
import ErrorMessage from '../components/common/ErrorMessage.vue'

use([ScatterChart, LineChart, GridComponent, TooltipComponent, LegendComponent, VisualMapComponent, GeoComponent, CanvasRenderer])

const loading = ref(true)
const error = ref(null)
const gridData = ref([])
const timeseriesData = ref([])
const cellDetail = ref(null)

// Filters
const selectedVariable = ref('evaporation_anomaly')
const selectedRegion = ref('')
const selectedYear = ref(null)
const selectedMonth = ref(null)

const variables = [
  { value: 'evaporation_anomaly', label: 'Evaporation Anomaly' },
  { value: 'temperature_anomaly', label: 'Temperature Anomaly' },
  { value: 'precipitation_anomaly', label: 'Precipitation Anomaly' },
  { value: 'radiation_anomaly', label: 'Radiation Anomaly' },
  { value: 'soil_moisture_anomaly', label: 'Soil Moisture Anomaly' },
]

const years = computed(() => [...new Set(gridData.value.map(d => d.year))].sort())
const months = computed(() => [...new Set(gridData.value.map(d => d.month))].sort())

async function loadData() {
  loading.value = true
  try {
    const params = { variable: selectedVariable.value, limit: 2000 }
    if (selectedRegion.value) params.region = selectedRegion.value
    if (selectedYear.value) params.year = selectedYear.value
    if (selectedMonth.value) params.month = selectedMonth.value

    const [grid, ts] = await Promise.all([
      fetchSpatialGrid(params),
      fetchTimeseries(selectedRegion.value ? { region: selectedRegion.value, variable: selectedVariable.value } : { variable: selectedVariable.value }),
    ])
    gridData.value = grid
    timeseriesData.value = ts
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

// Spatial heatmap (scatter plot with lat/lon)
const spatialChartOption = computed(() => {
  if (!gridData.value.length) return {}
  const sahara = gridData.value.filter(d => d.region === 'Sahara')
  const eastChina = gridData.value.filter(d => d.region === 'East China')

  const allValues = gridData.value.map(d => d.value).filter(v => v != null)
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)

  const series = []
  if (sahara.length) {
    series.push({
      name: 'Sahara',
      type: 'scatter',
      data: sahara.map(d => [d.longitude, d.latitude, d.value]),
      symbolSize: 8,
      itemStyle: { opacity: 0.7 },
    })
  }
  if (eastChina.length) {
    series.push({
      name: 'East China',
      type: 'scatter',
      data: eastChina.map(d => [d.longitude, d.latitude, d.value]),
      symbolSize: 8,
      itemStyle: { opacity: 0.7 },
    })
  }

  return {
    tooltip: {
      trigger: 'item',
      formatter: p => `${p.seriesName}<br/>Lon: ${p.value[0]?.toFixed(2)}<br/>Lat: ${p.value[1]?.toFixed(2)}<br/>${selectedVariable.value}: ${p.value[2]?.toFixed(4)}`,
    },
    legend: { top: 0, textStyle: { fontSize: 10 } },
    grid: { left: '3%', right: '4%', bottom: '8%', top: '12%', containLabel: true },
    visualMap: {
      min: minVal,
      max: maxVal,
      inRange: { color: ['#2563eb', '#f8fafc', '#ef4444'] },
      text: ['High', 'Low'],
      textStyle: { fontSize: 9 },
      left: 'right',
    },
    xAxis: { type: 'value', name: 'Longitude', nameTextStyle: { fontSize: 10 } },
    yAxis: { type: 'value', name: 'Latitude', nameTextStyle: { fontSize: 10 } },
    series,
  }
})

// Timeseries line chart
const timeseriesOption = computed(() => {
  if (!timeseriesData.value.length) return {}
  const regions = [...new Set(timeseriesData.value.map(d => d.region))]
  const timeLabels = timeseriesData.value.map(d => `${d.year}-${String(d.month).padStart(2, '0')}`)
  const uniqueTimes = [...new Set(timeLabels)].sort()

  const series = regions.map((region, i) => {
    const regionData = timeseriesData.value.filter(d => d.region === region)
    const timeMap = {}
    regionData.forEach(d => {
      const key = `${d.year}-${String(d.month).padStart(2, '0')}`
      timeMap[key] = d.evaporation_anomaly
    })
    const colors = ['#2563eb', '#0d9488']
    return {
      name: region,
      type: 'line',
      data: uniqueTimes.map(t => timeMap[t] ?? null),
      smooth: true,
      itemStyle: { color: colors[i % colors.length] },
    }
  })

  return {
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { fontSize: 10 } },
    grid: { left: '3%', right: '4%', bottom: '8%', top: '12%', containLabel: true },
    xAxis: { type: 'category', data: uniqueTimes, axisLabel: { fontSize: 8, rotate: 45, interval: 11 } },
    yAxis: { type: 'value', name: selectedVariable.value },
    series,
  }
})

// Cell click handler
function onChartClick(params) {
  if (params.componentType === 'series' && params.value) {
    const [lon, lat] = params.value
    fetchGridCellDetail({ latitude: lat, longitude: lon, year: 2015, month: 6 })
      .then(detail => { cellDetail.value = detail })
      .catch(() => { cellDetail.value = null })
  }
}

onMounted(loadData)
watch([selectedVariable, selectedRegion, selectedYear, selectedMonth], loadData)
</script>

<template>
  <div v-if="loading"><LoadingState message="Loading spatial data..." /></div>
  <div v-else-if="error"><ErrorMessage :message="error" /></div>
  <div v-else class="space-y-5">
    <!-- Filter Row -->
    <div class="card p-4 flex flex-wrap items-end gap-3">
      <div class="flex flex-col gap-1">
        <label class="text-[10px] font-semibold text-gray-500 uppercase">Variable</label>
        <select v-model="selectedVariable">
          <option v-for="v in variables" :key="v.value" :value="v.value">{{ v.label }}</option>
        </select>
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-[10px] font-semibold text-gray-500 uppercase">Region</label>
        <select v-model="selectedRegion">
          <option value="">All Regions</option>
          <option value="Sahara">Sahara</option>
          <option value="East China">East China</option>
        </select>
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-[10px] font-semibold text-gray-500 uppercase">Year</label>
        <select v-model="selectedYear">
          <option :value="null">All Years</option>
          <option v-for="y in years" :key="y" :value="y">{{ y }}</option>
        </select>
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-[10px] font-semibold text-gray-500 uppercase">Month</label>
        <select v-model="selectedMonth">
          <option :value="null">All Months</option>
          <option v-for="m in months" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>
    </div>

    <!-- Charts Row -->
    <div class="grid grid-cols-1 gap-5">
      <div class="card p-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-3">Spatial Grid — {{ selectedVariable }}</h4>
        <VChart :option="spatialChartOption" style="height:400px" autoresize @click="onChartClick" />
      </div>
    </div>
    <div class="card p-4">
      <h4 class="text-sm font-semibold text-gray-700 mb-3">Timeseries — {{ selectedVariable }}</h4>
      <VChart :option="timeseriesOption" style="height:300px" autoresize />
    </div>

    <!-- Cell Detail -->
    <div v-if="cellDetail" class="card p-4">
      <h4 class="text-sm font-semibold text-gray-800 mb-2">Grid Cell Detail</h4>
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
        <div><span class="text-gray-500">Region:</span> <strong>{{ cellDetail.region }}</strong></div>
        <div><span class="text-gray-500">Lat:</span> <strong>{{ cellDetail.latitude?.toFixed(4) }}</strong></div>
        <div><span class="text-gray-500">Lon:</span> <strong>{{ cellDetail.longitude?.toFixed(4) }}</strong></div>
        <div><span class="text-gray-500">Temperature:</span> <strong>{{ cellDetail.temperature?.toFixed(2) }} °C</strong></div>
        <div><span class="text-gray-500">Precipitation:</span> <strong>{{ cellDetail.precipitation?.toFixed(2) }} mm</strong></div>
        <div><span class="text-gray-500">Radiation:</span> <strong>{{ cellDetail.radiation?.toFixed(1) }} W/m²</strong></div>
        <div><span class="text-gray-500">Wind Speed:</span> <strong>{{ cellDetail.wind_speed?.toFixed(2) }} m/s</strong></div>
        <div><span class="text-gray-500">Evap Anomaly:</span> <strong>{{ cellDetail.evaporation_anomaly?.toFixed(4) }}</strong></div>
      </div>
    </div>
  </div>
</template>
