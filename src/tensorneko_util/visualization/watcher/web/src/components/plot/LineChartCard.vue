<script setup lang="ts">
import { BCard } from "bootstrap-vue-3";

import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { LineChart } from "echarts/charts";
import { DatasetComponent, GridComponent, LegendComponent, TitleComponent, TooltipComponent } from "echarts/components";
import VChart from "vue-echarts";
import { type LineChartData, lineCharts } from "@/data";
import type {
  DatasetComponentOption,
  EChartsOption,
  LegendComponentOption,
  SeriesOption,
  TitleComponentOption,
  TooltipComponentOption,
  XAXisComponentOption,
  YAXisComponentOption
} from "echarts";
import { onMounted, type Ref, ref } from "vue";
import type { CallbackDataParams, TopLevelFormatterParams } from "echarts/types/dist/shared";

use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DatasetComponent
]);

interface Props {
  k: string
}

const props = defineProps<Props>()
const option: Ref<EChartsOption> = ref({})
let d: LineChartData = lineCharts[props.k]

function update(): void {
  d = lineCharts[props.k];
  option.value = genOption(d);
}

onMounted(() => {
  update()
})

defineExpose({update})

function genOption(d: LineChartData): EChartsOption {
  const linesMap: Map<string, Array<{ x: number, y: number }>> = new Map();
  d.value.forEach(p => {
    if (!linesMap.has(p.label)) {
      linesMap.set(p.label, []);
    }
    linesMap.get(p.label)?.push({x: p.x, y: p.y});
  });
  const lines = Array.from(linesMap.entries());
  const lineKeys = lines.map(p => p[0]);

  const title: TitleComponentOption = {
    text: d.name,
    left: "center"
  };
  const tooltip: TooltipComponentOption = {
    trigger: "axis",
    axisPointer: {
      type: "line"
    },
    formatter: (params: TopLevelFormatterParams): string => {
      let [x, y] = (params as CallbackDataParams[])[0].data as [number, number];
      return `${d.x_label}: ${Math.round(x * 10000) / 10000}<br/>` +
          `${d.y_label}: ${Math.round(y * 10000) / 10000}`
    }
  };
  const legend: LegendComponentOption = {
    orient: "vertical",
    left: "left",
    data: lineKeys
  }
  const xAxis: XAXisComponentOption = {};
  const yAxis: YAXisComponentOption = {
    min: "dataMin"
  };
  const dataset: Array<DatasetComponentOption> = lines.map(([label, points]) => {
    return {
      name: label,
      source: points.map(p => [p.x, p.y]),
      dimensions: [d.x_label, d.y_label],
      encode: {
        x: d.x_label,
        y: d.y_label,
        tooltip: [d.x_label, d.y_label]
      }
    };
  });
  const series: Array<SeriesOption> = lineKeys.map((label, index) => {
    return {
      type: "line",
      name: label,
      datasetIndex: index
    };
  });

  return {title, tooltip, legend, xAxis, yAxis, dataset, series};
}

</script>

<template>
  <b-card class="w-25" style="max-width: 500px">
    <v-chart class="chart" :option="option"/>
  </b-card>
</template>

<style scoped>
.chart {
  height: 400px;
}
</style>
