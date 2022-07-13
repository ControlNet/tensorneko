<script setup lang="ts">
import { variables } from "@/data";
import VariableRow from "@/components/variable/VariableRow.vue";

const subs = new Map<string, any>()
defineExpose({subs})
</script>

<template>
  <h6 class="display-6">Variables</h6>
  <table class="table table-striped table-hover">
    <thead>
    <tr>
      <th scope="col" style="width: 200px;">Name</th>
      <th scope="col">Value</th>
    </tr>
    </thead>
    <tbody>
    <tr v-for="k of Object.keys(variables)" :key="k">
      <VariableRow :k="k" :ref="r => subs.set(k, r)"/>
    </tr>
    </tbody>
  </table>
</template>
