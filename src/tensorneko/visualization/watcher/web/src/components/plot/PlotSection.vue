<template>
  <h6 class="display-6">Plots</h6>
  <b-card-group>
    <LineChartCard v-for="k of Object.keys(lineCharts)" :key="k" :k="k" :ref="k"/>
  </b-card-group>
</template>

<script lang="ts">
import { Options, Vue } from "vue-class-component";
import LineChartCard from "@/components/plot/LineChartCard.vue";
import { lineCharts } from "@/data";

@Options({
  components: { LineChartCard }
})
export default class PlotSection extends Vue {
  lineCharts = lineCharts;

  $refs!: Record<string, LineChartCard>
}
</script>

<style scoped>

</style>