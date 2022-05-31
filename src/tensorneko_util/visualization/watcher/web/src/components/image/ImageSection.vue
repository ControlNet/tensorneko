<template>
  <h6 class="display-6">Images</h6>
  <b-card>
    <b-card-group v-for="[i, group] of Object.entries(imageGroups)" :key="i">
      <ImageCard v-for="k of Object.keys(group)" :key="k" :k="k" :ref="k"/>
    </b-card-group>
  </b-card>
</template>

<script lang="ts">
import { Options, Vue } from "vue-class-component";
import ImageCard from "@/components/image/ImageCard.vue";
import { Images, images } from "@/data";


@Options({
  components: { ImageCard }
})
export default class ImageSection extends Vue {
  images = images

  $refs!: Record<string, ImageCard>

  get imageGroups(): Array<Images> {
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

}
</script>

<style scoped>

</style>