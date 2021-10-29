<template>
  <div class="small">
    <v-chart class="chart" :option="this.conf"/>
  </div>
</template>

<script lang="ts">
import { Prop } from "vue-property-decorator";
import { LineChartData } from "@/data";
import { Ref, ref } from "vue";
import { Vue } from "vue-class-component";

export default class LineChart extends Vue {
  @Prop(LineChartData) chartData!: LineChartData;

  conf(): {options: Ref<unknown>} {
    return {
      options: ref({
        title: {

        }
      })
    }
  }

  mounted(): void {
  }
}

type Dataset = { label: string, backgroundColor: string, data: Array<number> }
</script>

<style scoped>

</style>