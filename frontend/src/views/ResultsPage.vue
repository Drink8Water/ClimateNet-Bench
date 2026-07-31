<script setup>
import { computed, ref } from 'vue'
import { use } from 'echarts/core'
import { BarChart, CustomChart, LineChart, ScatterChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import VChart from 'vue-echarts'
import { finalBenchmarkResults as results } from '../data/finalBenchmarkResults'
import { useI18n } from '../i18n'

use([
  BarChart,
  CustomChart,
  LineChart,
  ScatterChart,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  CanvasRenderer,
])

const { locale, pick, setLocale } = useI18n()
const selectedMetric = ref('rmse')
const copied = ref(false)

const colors = {
  ink: '#17212b',
  muted: '#65717a',
  rule: '#d9ddde',
  blue: '#1769a5',
  red: '#c45137',
  paper: '#f7f7f4',
}

const metricDefinitions = {
  rmse: {
    label: 'RMSE',
    mean: 'rmseMean',
    std: 'rmseStd',
    min: 0,
    max: 12,
    lowerIsBetter: true,
  },
  mae: {
    label: 'MAE',
    mean: 'maeMean',
    std: 'maeStd',
    min: 0,
    max: 7,
    lowerIsBetter: true,
  },
  r2: {
    label: 'R²',
    mean: 'r2Mean',
    std: 'r2Std',
    min: 0,
    max: 0.8,
    lowerIsBetter: false,
  },
  skill: {
    label: pick('相对气候态 Skill', 'Skill vs climatology'),
    mean: 'skillMean',
    std: 'skillStd',
    min: 0,
    max: 0.55,
    lowerIsBetter: false,
  },
}

const protocolLabels = computed(() => [
  pick('Random · 3 seeds', 'Random · 3 seeds'),
  pick('Temporal · 3 seeds', 'Temporal · 3 seeds'),
  pick('Repeated spatial · 5 folds', 'Repeated spatial · 5 folds'),
])

const formalRows = computed(() => {
  const rows = results.multiSeedMetrics.filter(row => ['random', 'temporal'].includes(row.split))
  return [
    ...rows,
    ...results.repeatedSpatialMetrics.map(row => ({ ...row, split: 'repeated spatial' })),
  ]
})

function formatMetric(value) {
  return Number(value).toFixed(3)
}

function forestRenderer(color, offset) {
  return (params, api) => {
    const mean = api.value(0)
    const std = api.value(1)
    const category = api.value(2)
    const center = api.coord([mean, category])
    const start = api.coord([mean - std, category])
    const end = api.coord([mean + std, category])
    const y = center[1] + offset
    const labelX = Math.min(center[0] + 10, params.coordSys.x + params.coordSys.width - 82)
    const compact = params.coordSys.width < 350

    return {
      type: 'group',
      children: [
        {
          type: 'line',
          shape: { x1: start[0], y1: y, x2: end[0], y2: y },
          style: { stroke: color, lineWidth: 1.5 },
        },
        {
          type: 'line',
          shape: { x1: start[0], y1: y - 4, x2: start[0], y2: y + 4 },
          style: { stroke: color, lineWidth: 1 },
        },
        {
          type: 'line',
          shape: { x1: end[0], y1: y - 4, x2: end[0], y2: y + 4 },
          style: { stroke: color, lineWidth: 1 },
        },
        {
          type: 'circle',
          shape: { cx: center[0], cy: y, r: 4.5 },
          style: { fill: colors.paper, stroke: color, lineWidth: 2 },
        },
        !compact && {
          type: 'text',
          style: {
            x: labelX,
            y,
            text: `${formatMetric(mean)} ± ${formatMetric(std)}`,
            fill: color,
            font: '600 10px SFMono-Regular, Consolas, monospace',
            verticalAlign: 'middle',
          },
        },
      ].filter(Boolean),
    }
  }
}

const forestOption = computed(() => {
  const metric = metricDefinitions[selectedMetric.value]
  const splitIndex = { random: 0, temporal: 1, 'repeated spatial': 2 }
  const forModel = model => formalRows.value
    .filter(row => row.model === model)
    .map(row => [row[metric.mean], row[metric.std], splitIndex[row.split], row.model, row.split])

  const baseOption = {
    animationDuration: 300,
    animationDurationUpdate: 300,
    grid: { left: 164, right: 92, top: 18, bottom: 42 },
    tooltip: {
      trigger: 'item',
      backgroundColor: '#17212b',
      borderWidth: 0,
      textStyle: { color: '#f7f7f4', fontSize: 11 },
      formatter: params => {
        const [mean, std, , model, split] = params.data
        return `${model}<br>${split}<br>${metric.label}: ${formatMetric(mean)} ± ${formatMetric(std)}`
      },
    },
    xAxis: {
      type: 'value',
      min: metric.min,
      max: metric.max,
      name: `${metric.label} · ${metric.lowerIsBetter ? pick('越低越好', 'lower is better') : pick('越高越好', 'higher is better')}`,
      nameLocation: 'end',
      nameGap: 12,
      nameTextStyle: { color: colors.muted, fontSize: 10 },
      axisLine: { lineStyle: { color: colors.ink, width: 1 } },
      axisTick: { show: false },
      axisLabel: { color: colors.muted, fontFamily: 'monospace', fontSize: 10 },
      splitLine: { lineStyle: { color: colors.rule, width: 1 } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: protocolLabels.value,
      axisLine: { lineStyle: { color: colors.ink, width: 1 } },
      axisTick: { show: false },
      axisLabel: {
        width: 146,
        align: 'left',
        color: colors.ink,
        fontSize: 11,
        fontWeight: 650,
      },
      splitLine: { show: true, lineStyle: { color: colors.rule } },
    },
    series: [
      {
        name: 'LightGBM',
        type: 'custom',
        renderItem: forestRenderer(colors.blue, -13),
        encode: { x: [0, 1], y: 2 },
        data: forModel('LightGBM'),
      },
      {
        name: 'Linear Regression',
        type: 'custom',
        renderItem: forestRenderer(colors.red, 13),
        encode: { x: [0, 1], y: 2 },
        data: forModel('Linear Regression'),
      },
    ],
  }
  return {
    baseOption,
    media: [{
      query: { maxWidth: 520 },
      option: {
        grid: { left: 126, right: 12, top: 18, bottom: 42 },
        yAxis: {
          axisLabel: {
            width: 112,
            fontSize: 9,
          },
        },
      },
    }],
  }
})

const featureOption = computed(() => {
  const splits = ['random', 'temporal', 'spatial']
  const values = (model, features) => splits.map(split => (
    results.correctedV1Metrics.find(row => row.model === model && row.features === features && row.split === split)?.rmse
  ))
  return {
    animationDuration: 350,
    grid: { left: 48, right: 20, top: 30, bottom: 38 },
    tooltip: { trigger: 'axis' },
    legend: {
      top: 0,
      right: 0,
      itemWidth: 18,
      itemHeight: 2,
      textStyle: { color: colors.muted, fontSize: 9 },
    },
    xAxis: {
      type: 'category',
      data: ['Random', 'Temporal', 'Spatial'],
      axisLine: { lineStyle: { color: colors.ink } },
      axisTick: { show: false },
      axisLabel: { color: colors.muted, fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      name: 'RMSE',
      min: 4,
      max: 13,
      nameTextStyle: { color: colors.muted, fontSize: 9 },
      axisLine: { show: true, lineStyle: { color: colors.ink } },
      axisTick: { show: false },
      axisLabel: { color: colors.muted, fontFamily: 'monospace', fontSize: 9 },
      splitLine: { lineStyle: { color: colors.rule } },
    },
    series: [
      { name: 'LightGBM full', type: 'line', data: values('LightGBM', 'full'), symbol: 'circle', symbolSize: 7, itemStyle: { color: colors.blue }, lineStyle: { color: colors.blue, width: 2 } },
      { name: 'LightGBM base', type: 'line', data: values('LightGBM', 'base'), symbol: 'emptyCircle', symbolSize: 7, itemStyle: { color: colors.blue }, lineStyle: { color: colors.blue, width: 1, type: 'dashed' } },
      { name: 'Linear full', type: 'line', data: values('Linear Regression', 'full'), symbol: 'circle', symbolSize: 7, itemStyle: { color: colors.red }, lineStyle: { color: colors.red, width: 2 } },
      { name: 'Linear base', type: 'line', data: values('Linear Regression', 'base'), symbol: 'emptyCircle', symbolSize: 7, itemStyle: { color: colors.red }, lineStyle: { color: colors.red, width: 1, type: 'dashed' } },
    ],
  }
})

const regionalOption = computed(() => ({
  animationDuration: 350,
  grid: { left: 46, right: 22, top: 30, bottom: 38 },
  tooltip: { trigger: 'axis' },
  legend: {
    top: 0,
    right: 0,
    itemWidth: 18,
    itemHeight: 2,
    textStyle: { color: colors.muted, fontSize: 9 },
  },
  xAxis: {
    type: 'category',
    data: ['Random', 'Spatial', 'Temporal'],
    axisLine: { lineStyle: { color: colors.ink } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontSize: 10 },
  },
  yAxis: {
    type: 'value',
    name: 'RMSE',
    min: 0,
    max: 14,
    nameTextStyle: { color: colors.muted, fontSize: 9 },
    axisLine: { show: true, lineStyle: { color: colors.ink } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: 'monospace', fontSize: 9 },
    splitLine: { lineStyle: { color: colors.rule } },
  },
  series: [
    {
      name: 'East China',
      type: 'line',
      data: results.regionalMetrics.map(row => row.eastChinaRmse),
      symbolSize: 8,
      itemStyle: { color: colors.red },
      lineStyle: { color: colors.red, width: 2 },
      label: { show: true, position: 'top', color: colors.red, fontFamily: 'monospace', fontSize: 9, formatter: p => p.value.toFixed(3) },
    },
    {
      name: 'Sahara',
      type: 'line',
      data: results.regionalMetrics.map(row => row.saharaRmse),
      symbolSize: 8,
      itemStyle: { color: colors.blue },
      lineStyle: { color: colors.blue, width: 2 },
      label: { show: true, position: 'bottom', color: colors.blue, fontFamily: 'monospace', fontSize: 9, formatter: p => p.value.toFixed(3) },
    },
  ],
}))

const compositionOption = computed(() => ({
  animationDuration: 350,
  grid: { left: 52, right: 28, top: 20, bottom: 42 },
  tooltip: {
    formatter: params => {
      const row = results.singleSpatialComposition.find(item => item.seed === params.data[3])
      return [
        `Seed ${row.seed}`,
        `East China: ${row.eastChinaShare}%`,
        `LightGBM RMSE: ${row.lightgbmRmse.toFixed(3)}`,
        `Target std: ${row.targetStd.toFixed(3)}`,
        `Test grids: ${row.testGrids.toLocaleString('en-US')}`,
      ].join('<br>')
    },
  },
  xAxis: {
    type: 'value',
    name: 'East China test share (%)',
    min: 0,
    max: 40,
    nameLocation: 'middle',
    nameGap: 28,
    nameTextStyle: { color: colors.muted, fontSize: 9 },
    axisLine: { lineStyle: { color: colors.ink } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: 'monospace', fontSize: 9 },
    splitLine: { lineStyle: { color: colors.rule } },
  },
  yAxis: {
    type: 'value',
    name: 'LightGBM RMSE',
    min: 5,
    max: 8.5,
    nameTextStyle: { color: colors.muted, fontSize: 9 },
    axisLine: { show: true, lineStyle: { color: colors.ink } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: 'monospace', fontSize: 9 },
    splitLine: { lineStyle: { color: colors.rule } },
  },
  series: [{
    type: 'scatter',
    data: results.singleSpatialComposition.map(row => [row.eastChinaShare, row.lightgbmRmse, row.targetStd, row.seed]),
    symbolSize: data => 11 + (data[2] - 10) * 4,
    itemStyle: { color: colors.blue, borderColor: colors.paper, borderWidth: 2 },
    label: {
      show: true,
      position: 'top',
      color: colors.ink,
      fontSize: 9,
      formatter: params => `seed ${params.data[3]}`,
    },
  }],
}))

const driverOption = computed(() => ({
  animationDuration: 350,
  grid: { left: 154, right: 42, top: 14, bottom: 34 },
  tooltip: { trigger: 'axis', axisPointer: { type: 'none' } },
  xAxis: {
    type: 'value',
    min: 0,
    max: 1,
    name: 'Pearson r',
    nameLocation: 'end',
    nameGap: 10,
    nameTextStyle: { color: colors.muted, fontSize: 9 },
    axisLine: { lineStyle: { color: colors.ink } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: 'monospace', fontSize: 9 },
    splitLine: { lineStyle: { color: colors.rule } },
  },
  yAxis: {
    type: 'category',
    inverse: true,
    data: results.foldDifficultyDrivers.map(row => pick(row.labelZh, row.labelEn)),
    axisLine: { lineStyle: { color: colors.ink } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontSize: 9, align: 'left', width: 142 },
  },
  series: [
    {
      type: 'bar',
      data: results.foldDifficultyDrivers.map(row => row.correlation),
      barWidth: 2,
      itemStyle: { color: colors.blue },
    },
    {
      type: 'scatter',
      data: results.foldDifficultyDrivers.map((row, index) => [row.correlation, index]),
      symbolSize: 8,
      itemStyle: { color: colors.paper, borderColor: colors.blue, borderWidth: 2 },
      label: {
        show: true,
        position: 'right',
        color: colors.blue,
        fontFamily: 'monospace',
        fontSize: 9,
        formatter: params => `r = ${params.value[0].toFixed(2)}`,
      },
    },
  ],
}))

const correctionOption = computed(() => ({
  animationDuration: 350,
  grid: { left: 68, right: 34, top: 24, bottom: 38 },
  tooltip: { trigger: 'axis' },
  xAxis: {
    type: 'category',
    data: [pick('失效源数据', 'Invalid source'), pick('Corrected 数据', 'Corrected data')],
    axisLine: { lineStyle: { color: colors.ink } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontSize: 9 },
  },
  yAxis: {
    type: 'value',
    name: 'Temporal RMSE',
    min: 5,
    max: 13,
    nameTextStyle: { color: colors.muted, fontSize: 9 },
    axisLine: { show: true, lineStyle: { color: colors.ink } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: 'monospace', fontSize: 9 },
    splitLine: { lineStyle: { color: colors.rule } },
  },
  series: results.sourceCorrectionAudit.map((row, index) => ({
    name: row.model,
    type: 'line',
    data: [
      { value: row.invalid, itemStyle: { color: colors.red } },
      { value: row.corrected, itemStyle: { color: colors.blue } },
    ],
    symbolSize: 8,
    lineStyle: { color: index === 0 ? colors.blue : colors.muted, width: 1.5 },
    label: {
      show: true,
      position: index === 0 ? 'top' : 'bottom',
      color: colors.ink,
      fontFamily: 'monospace',
      fontSize: 9,
      formatter: params => `${row.model === 'LightGBM' ? 'LightGBM' : 'Linear'} ${params.value.toFixed(3)}`,
    },
  })),
}))

const formalClaims = computed(() => [
  { value: '27.8%', zh: 'Repeated spatial RMSE 相对 Linear 的降幅', en: 'Repeated spatial RMSE reduction vs Linear' },
  { value: '5 / 5', zh: 'LightGBM 在所有空间折中胜出', en: 'LightGBM wins every spatial fold' },
  { value: '18 / 18', zh: 'East China RMSE 高于 Sahara 的区域比较', en: 'Regional comparisons where East China RMSE is higher' },
  { value: '28 / 28', zh: '正式评测任务完成，零失败', en: 'Formal evaluation tasks complete, zero failures' },
])

function downloadResults() {
  const header = ['model', 'protocol', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std', 'skill_mean', 'skill_std']
  const rows = formalRows.value.map(row => [
    row.model,
    row.split,
    row.maeMean,
    row.maeStd,
    row.rmseMean,
    row.rmseStd,
    row.r2Mean,
    row.r2Std,
    row.skillMean,
    row.skillStd,
  ])
  const csv = [header, ...rows].map(row => row.join(',')).join('\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = 'climatenet_corrected_formal_results.csv'
  link.click()
  URL.revokeObjectURL(url)
}

async function copyCommand() {
  try {
    await navigator.clipboard.writeText(results.reproduction.command)
    copied.value = true
    window.setTimeout(() => { copied.value = false }, 1600)
  } catch {
    copied.value = false
  }
}
</script>

<template>
  <div class="research-page">
    <a class="skip-link" href="#results">{{ pick('跳到正式结果', 'Skip to formal results') }}</a>

    <header class="paper-header">
      <div>
        <p class="brand-line">ClimateNet-Bench / Corrected ERA5-Land</p>
        <h1>{{ pick('下一月蒸发异常预测：正式实验结果', 'Next-month evaporation anomaly forecasting: formal results') }}</h1>
        <p class="task-line">
          {{ pick(
            '输入过去 6 个月气候变量，预测 month t 的 evaporation_anomaly',
            'Use the previous six months of climate variables to predict evaporation_anomaly at month t',
          ) }}
        </p>
      </div>
      <div class="header-actions">
        <div class="locale-switch" aria-label="language">
          <button :class="{ active: locale === 'zh' }" type="button" @click="setLocale('zh')">中文</button>
          <button :class="{ active: locale === 'en' }" type="button" @click="setLocale('en')">EN</button>
        </div>
        <button class="text-action" type="button" @click="downloadResults">
          {{ pick('下载结果 CSV', 'Download results CSV') }}
        </button>
        <a class="text-action accent" href="#reproduce">{{ pick('复现实验', 'Reproduce') }}</a>
      </div>
    </header>

    <div class="study-strip" aria-label="benchmark scope">
      <span>REGIONS <strong>{{ results.scope.regions.join(', ') }}</strong></span>
      <span>PERIOD <strong>{{ results.scope.period }}</strong></span>
      <span>SAMPLES <strong>{{ results.dataset.lagSamples.toLocaleString('en-US') }}</strong></span>
      <span>MODELS <strong>Linear, LightGBM</strong></span>
      <span>PROTOCOL <strong>3 seeds, 5 spatial folds</strong></span>
      <span class="verified">28 / 28 COMPLETE · ZERO GRID LEAKAGE</span>
    </div>

    <main id="results">
      <section class="lead-results">
        <div class="lead-figure">
          <div class="figure-heading">
            <div>
              <p class="figure-index">Figure 1</p>
              <h2>{{ pick('不同泛化协议下的预测表现', 'Predictive performance across generalisation protocols') }}</h2>
              <p>{{ pick('点为均值，线段为 mean ± sample standard deviation。', 'Points show means; whiskers show mean ± sample standard deviation.') }}</p>
            </div>
            <div class="metric-switch" :aria-label="pick('选择指标', 'Choose metric')">
              <button
                v-for="metric in ['rmse', 'mae', 'r2', 'skill']"
                :key="metric"
                :class="{ active: selectedMetric === metric }"
                type="button"
                @click="selectedMetric = metric"
              >
                {{ metricDefinitions[metric].label }}
              </button>
            </div>
          </div>
          <div class="chart-legend" aria-hidden="true">
            <span class="lightgbm">LightGBM</span>
            <span class="linear">Linear Regression</span>
          </div>
          <div class="forest-scroll">
            <VChart class="forest-chart" :option="forestOption" autoresize />
          </div>
        </div>

        <aside class="claim-column">
          <p class="claim-heading">{{ pick('可支持的结论', 'Supported conclusions') }}</p>
          <article v-for="claim in formalClaims" :key="claim.value" class="claim-row">
            <strong>{{ claim.value }}</strong>
            <p>{{ pick(claim.zh, claim.en) }}</p>
          </article>
          <p class="scope-boundary">
            <b>EXCLUDED</b>
            {{ pick(
              'source_data_invalid 历史运行与 synthetic smoke/sample 结果不参与均值、排名和结论。',
              'Historical source_data_invalid runs and synthetic smoke/sample results are excluded from aggregates, rankings, and claims.',
            ) }}
          </p>
        </aside>
      </section>

      <section class="paired-figures">
        <article class="figure-block">
          <p class="figure-index">Figure 2</p>
          <h2>{{ pick('Base 与 Full 特征配置', 'Base and full feature configurations') }}</h2>
          <p>{{ pick('Corrected v1 单次参考运行。实线为 full，虚线为 base。', 'Corrected v1 reference run. Solid lines are full; dashed lines are base.') }}</p>
          <VChart class="medium-chart" :option="featureOption" autoresize />
        </article>

        <article class="figure-block">
          <p class="figure-index">Figure 3</p>
          <h2>{{ pick('区域误差热点', 'Regional error hotspot') }}</h2>
          <p>{{ pick('Corrected v1 LightGBM full。East China 在所有协议下误差更高。', 'Corrected v1 LightGBM full. East China has higher error under every protocol.') }}</p>
          <VChart class="medium-chart" :option="regionalOption" autoresize />
        </article>
      </section>

      <section class="diagnostic-band">
        <div class="band-heading">
          <p class="figure-index">Figure 4</p>
          <h2>{{ pick('单次空间划分为何不稳定', 'Why one spatial holdout is unstable') }}</h2>
          <p>
            {{ pick(
              '气泡大小表示 target standard deviation。Seed 2026 的测试集仅含 3.9% East China，因此明显更容易。',
              'Bubble size represents target standard deviation. Seed 2026 contains only 3.9% East China and is materially easier.',
            ) }}
          </p>
        </div>
        <VChart class="wide-chart" :option="compositionOption" autoresize />
        <div class="composition-table" role="table" :aria-label="pick('空间划分组成', 'Spatial split composition')">
          <div class="composition-row composition-head" role="row">
            <span>Seed</span>
            <span>Test samples</span>
            <span>Test grids</span>
            <span>East China</span>
            <span>Target std</span>
            <span>Zero baseline</span>
            <span>Persistence</span>
            <span>Feature SMD</span>
            <span>LightGBM RMSE</span>
          </div>
          <div v-for="row in results.singleSpatialComposition" :key="row.seed" class="composition-row" role="row">
            <strong>{{ row.seed }}</strong>
            <span>{{ row.testSamples.toLocaleString('en-US') }}</span>
            <span>{{ row.testGrids.toLocaleString('en-US') }}</span>
            <span>{{ row.eastChinaShare.toFixed(1) }}%</span>
            <span>{{ row.targetStd.toFixed(3) }}</span>
            <span>{{ row.zeroBaselineRmse.toFixed(3) }}</span>
            <span>{{ row.persistenceRmse.toFixed(3) }}</span>
            <span>{{ row.meanFeatureSmd.toFixed(3) }}</span>
            <span>{{ row.lightgbmRmse.toFixed(3) }}</span>
          </div>
        </div>
      </section>

      <section class="paired-figures evidence-pair">
        <article class="figure-block">
          <p class="figure-index">Figure 5</p>
          <h2>{{ pick('Repeated spatial fold 难度', 'Repeated spatial fold difficulty') }}</h2>
          <p>{{ pick('Pearson r，n=5，仅作描述性解释，不作统计推断。', 'Pearson r, n=5. Descriptive only, not inferential.') }}</p>
          <VChart class="small-chart" :option="driverOption" autoresize />
          <div class="fold-range">
            <span><b>Fold 0</b> 8.871 RMSE · 33.2% East China · {{ pick('最难', 'hardest') }}</span>
            <span><b>Fold 4</b> 6.212 RMSE · {{ pick('最容易', 'easiest') }}</span>
          </div>
        </article>

        <article class="figure-block audit-figure">
          <p class="figure-index">Figure 6</p>
          <h2>{{ pick('数据修正审计', 'Source correction audit') }}</h2>
          <p>{{ pick('旧值仅用于解释失效原因，不属于正式模型结果。', 'Old values explain the failure mode only and are not formal model results.') }}</p>
          <VChart class="small-chart" :option="correctionOption" autoresize />
          <p class="audit-note">
            {{ pick(
              'LightGBM temporal collapse 由累计变量源数据问题造成。Corrected 数据恢复了稳定结果。',
              'The LightGBM temporal collapse came from the accumulated-variable source issue. Corrected data restores stable performance.',
            ) }}
          </p>
        </article>
      </section>

      <section id="reproduce" class="reproduction">
        <div>
          <p class="figure-index">Reproducibility</p>
          <h2>{{ pick('复现正式空间评测', 'Reproduce the formal spatial benchmark') }}</h2>
          <dl>
            <dt>Data</dt>
            <dd>ERA5-Land corrected · SHA256 verified</dd>
            <dt>Config</dt>
            <dd>{{ results.reproduction.config }}</dd>
            <dt>Outputs</dt>
            <dd>{{ results.reproduction.outputs.join(' · ') }}</dd>
            <dt>Preprocess</dt>
            <dd>per-split train-only</dd>
          </dl>
        </div>
        <div class="command-block">
          <code>{{ results.reproduction.command }}</code>
          <button type="button" @click="copyCommand">
            {{ copied ? pick('已复制', 'Copied') : pick('复制命令', 'Copy command') }}
          </button>
          <p>5 folds · 2 models · 10 tasks · 0 failures</p>
        </div>
      </section>
    </main>

    <footer>
      <span>ClimateNet-Bench</span>
      <span>{{ pick('结论范围：Sahara 与 East China，2019-2023', 'Scope: Sahara and East China, 2019-2023') }}</span>
    </footer>
  </div>
</template>
