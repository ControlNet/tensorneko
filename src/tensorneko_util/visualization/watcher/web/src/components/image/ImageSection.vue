<script setup lang="ts">
import ImageCard from "@/components/image/ImageCard.vue";
import type { Images } from "@/data";
import { BCard, BCardGroup } from "bootstrap-vue-3";
import { images } from "@/data";

const subs = new Map<string, any>()
defineExpose({subs})

function imageGroups(): Array<Images> {
  const groups: Array<Images> = [{}];
  let index = 0;
  let group_ind = 0;
  for (const [k, image] of Object.entries(images)) {
    if (index !== 4) {
      groups[group_ind][k] = image;
      index++;
    } else {
      index = 1;
      group_ind++;
      eval(`groups.push({${k}: image})`);
    }
  }
  return groups;
}

</script>


<template>
  <h6 class="display-6">Images</h6>
  <b-card>
    <b-card-group v-for="[i, group] of Object.entries(imageGroups())" :key="i">
      <ImageCard v-for="k of Object.keys(group)" :key="k" :k="k" :ref="r => subs.set(k, r)"/>
    </b-card-group>
  </b-card>
</template>
