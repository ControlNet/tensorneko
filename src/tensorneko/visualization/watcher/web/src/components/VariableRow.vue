<template>
  <th scope="row">{{ d.name }}</th>
  <td>{{ d.value }}</td>
</template>

<script lang="ts">
import { Vue } from 'vue-class-component';
import { variables } from "@/data";
import { Prop } from "vue-property-decorator";


export default class VariableRow extends Vue {
  @Prop(String) k!: string

  d = variables[this.k]

  update(): void {
    this.d = variables[this.k];
  }
}
</script>
