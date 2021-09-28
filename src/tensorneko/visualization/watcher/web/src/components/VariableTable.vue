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
      <VariableRow :k="k" :ref="k"/>
    </tr>
    </tbody>
  </table>
</template>

<script lang="ts">
import { Options, Vue } from "vue-class-component";
import { variables } from "@/data";
import VariableRow from "@/components/VariableRow.vue";

@Options({
  components: { VariableRow }
})
export default class VariableTable extends Vue {
  variables = variables

  get refs(): Record<string, VariableRow> {
    return this.$refs as Record<string, VariableRow>
  }
}
</script>
