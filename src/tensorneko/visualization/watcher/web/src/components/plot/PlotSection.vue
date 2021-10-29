<template>

</template>

<script lang="ts">
import { Vue, Options } from "vue-class-component";
import LineChart from "@/components/plot/LineChart.vue";

@Options({
    components: { LineChart }
})
export default class PlotSection extends Vue {
}
</script>

<style scoped>

</style>