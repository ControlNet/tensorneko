<script setup lang="ts">
import LineChartCard from "@/components/plot/LineChartCard.vue";
import { lineCharts } from "@/data";
import { BCardGroup } from "bootstrap-vue-3";

const subs = new Map<string, any>()
defineExpose({subs})
</script>

<template>
  <h6 class="display-6">Plots</h6>
  <b-card-group>
    <LineChartCard v-for="k of Object.keys(lineCharts)" :key="k" :k="k" :ref="r => subs.set(k, r)"/>
  </b-card-group>
</template>
