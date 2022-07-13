<script setup lang="ts">

import { BCard, BCardTitle } from "bootstrap-vue-3";
import { defineExpose, onMounted, type Ref, ref } from "vue";
import { images } from "@/data";

interface Props {
  k: string
}
const props = defineProps<Props>()
const path: Ref<string> = ref("")

function update(): void {
  path.value = images[props.k].value
}

onMounted(() => update())
defineExpose({update})

</script>

<template>
  <b-card :img-src="path" img-alt="NULL" class="image-card m-1" img-top>
    <b-card-title>{{k}}</b-card-title>
  </b-card>
</template>

<style scoped>

.image-card {
  max-width: 256px;
  text-align: center;
}

</style>
