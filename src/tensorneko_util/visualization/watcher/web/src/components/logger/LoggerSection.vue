<template>
  <h6 class="display-6">Logs</h6>
  <LoggerCard v-for="k of Object.keys(logs)" :key="k" :k="k" :ref="k"/>
</template>

<script lang="ts">
import { Options, Vue } from "vue-class-component";
import { logs } from "@/data";
import LoggerCard from "@/components/logger/LoggerCard.vue";

@Options({
  components: { LoggerCard }
})
export default class LoggerSection extends Vue {
  readonly logs = logs

  $refs!: Record<string, LoggerCard>

}
</script>

<style scoped>

</style>