<template>
  <b-card class="w-25" style="max-width: 500px">
    <v-chart class="chart" :option="option"/>
  </b-card>
</template>

<script lang="ts">
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { LineChart } from "echarts/charts";
import { TitleComponent, TooltipComponent, LegendComponent, GridComponent, DatasetComponent } from "echarts/components";
import VChart from "vue-echarts";
import { Options, Vue } from "vue-class-component";
import { Prop } from "vue-property-decorator";
import { LineChartData, lineCharts } from "@/data";
import {
  DatasetComponentOption,
  EChartsOption, LegendComponentOption,
  SeriesOption, TitleComponentOption,
  TooltipComponentOption,
  XAXisComponentOption,
  YAXisComponentOption
} from "echarts";
import { CallbackDataParams } from "echarts/types/dist/shared";

use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DatasetComponent
]);

@Options({
  name: "LineChartCard",
  components: { VChart }
})
export default class LineChartCard extends Vue {

  @Prop(String) k!: string;

  d: LineChartData = lineCharts[this.k];

  update(): void {
    this.d = lineCharts[this.k];
    this.option = this.genOption(this.d);
  }

  xLabel = this.d.x_label;
  yLabel = this.d.y_label;

  option: EChartsOption = this.genOption(lineCharts[this.k]);

  genOption(d: LineChartData): EChartsOption {
    const linesMap: Map<string, Array<{ x: number, y: number }>> = new Map();
    d.value.forEach(p => {
      if (!linesMap.has(p.label)) {
        linesMap.set(p.label, []);
      }
      linesMap.get(p.label).push({x: p.x, y: p.y});
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
      formatter: (params: CallbackDataParams) => {
        return `${this.xLabel}: ${Math.round(params[0].data[0] * 10000) / 10000}<br/>` +
            `${this.yLabel}: ${Math.round(params[0].data[1] * 10000) / 10000}`
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
        dimensions: [this.xLabel, this.yLabel],
        encode: {
          x: this.xLabel,
          y: this.yLabel,
          tooltip: [this.xLabel, this.yLabel]
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

    return { title, tooltip, legend, xAxis, yAxis, dataset, series };
  }
}
</script>

<style scoped>
.chart {
  height: 400px;
}
</style>