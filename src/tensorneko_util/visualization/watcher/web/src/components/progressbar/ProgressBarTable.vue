<script lang="ts" setup>
import { progressbars } from "@/data";
import ProgressBarRow from "@/components/progressbar/ProgressBarRow.vue";

const subs = new Map<string, any>()
defineExpose({subs})
</script>

<template>
  <h6 class="display-6">ProgressBars</h6>
  <table class="table table-striped table-hover">
    <thead>
    <tr>
      <th scope="col" style="width: 200px;">Name</th>
      <th scope="col"/>
    </tr>
    </thead>
    <tbody>
    <tr v-for="k of Object.keys(progressbars)" :key="k" class="table-secondary">
      <ProgressBarRow :k="k" :ref="r => subs.set(k, r)"/>
    </tr>
    </tbody>
  </table>
</template>
