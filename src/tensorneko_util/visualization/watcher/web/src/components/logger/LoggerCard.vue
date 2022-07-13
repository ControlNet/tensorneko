<script lang="ts" setup>
import { defineExpose, type Ref, ref } from "vue";
import { logs } from "@/data";
import { BCard, BCardText } from "bootstrap-vue-3";

interface Props {
  k: string
}

const props = defineProps<Props>()
const content: Ref<Array<string>> = ref(logs[props.k].value)

function update(): void {
  content.value = logs[props.k].value
}

defineExpose({update})

</script>

<template>
  <b-card>
    <b-card :title="k">
      <b-card-text v-for="text of content" :key="text">{{ text }}</b-card-text>
    </b-card>
  </b-card>
</template>
