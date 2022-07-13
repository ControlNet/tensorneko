<script setup lang="ts">
import { type VariableData, variables } from "@/data";
import { onMounted, ref } from 'vue';

interface Props {
  k: string
}
const props = defineProps<Props>()

const name = ref("")
const value = ref("")
let d: VariableData | null = null

function update(): void {
  d = variables[props.k];
  name.value = d.name
  value.value = d.value
}

onMounted(() => {
  update()
})

defineExpose({update})

</script>

<template>
  <th scope="row">{{ name }}</th>
  <td>{{ value }}</td>
</template>

