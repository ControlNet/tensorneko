<script lang="ts" setup>
import type { ProgressBarData } from "@/data";
import { computed, onMounted, ref } from "vue";
import type { Ref } from "vue";
import { progressbars } from "@/data";
import { BProgress, BProgressBar } from "bootstrap-vue-3";

let d: Ref<ProgressBarData | null> = ref(null)
let prev: ProgressBarData | null = null
let lastModified: number = +new Date()
const variant: Ref<VariantType> = ref("warning")

interface Props {
  k: string
}

const props = defineProps<Props>()

let preValue: number | null = null

type VariantType = "success" | "warning" | "danger";

function getType(): VariantType {
  const dTime = +new Date() - lastModified;

  if (dTime > 10000) {
    return "warning";
  } else if ((prev?.value ?? null) !== null) {
    if (value.value > (preValue ?? 0)) {
      return "success";
    } else if (value.value < (preValue ?? 0)) {
      return "danger";
    } else {
      return "warning";
    }
  } else {
    return "warning";
  }
}

function updateVariant(): void {
  variant.value = getType()
  setTimeout(updateVariant, 10000)
}

function update(): void {
  prev = d.value;
  d.value = progressbars[props.k]
  preValue = prev?.value ?? 0
  lastModified = +new Date()
}

onMounted(() => {
  update()
  setTimeout(updateVariant, 2000)
})

const total = computed(() => d.value?.total)
const name = computed(() => d.value?.name ?? "")
const value = computed(() => d.value?.value ?? 0)
defineExpose({update})

</script>


<template>
  <th scope="row">{{ name }}</th>
  <td>
    <b-progress striped animated style="height: 30px;">
      <b-progress-bar :value="value" :variant="variant" :max="total">
        <span><strong>
          {{ value }} / {{ total }}
        </strong></span>
      </b-progress-bar>
    </b-progress>
  </td>
</template>

<style scoped>
div.progress-bar > span {
  color: black;
}
</style>
