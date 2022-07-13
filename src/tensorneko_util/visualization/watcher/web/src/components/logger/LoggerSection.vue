<script setup lang="ts">
import { logs } from "@/data";
import LoggerCard from "@/components/logger/LoggerCard.vue";

const subs = new Map<string, any>()
defineExpose({subs})
</script>

<template>
  <h6 class="display-6">Logs</h6>
  <LoggerCard v-for="k of Object.keys(logs)" :key="k" :k="k" :ref="r => subs.set(k, r)"/>
</template>

<style scoped>

</style>
