<template>
  <b-card :img-src="path" img-alt="NULL" class="image-card m-1" img-top>
    <b-card-title>{{k}}</b-card-title>
  </b-card>
</template>

<script lang="ts">
import { Vue } from "vue-class-component";
import { Prop } from "vue-property-decorator";
import { images } from "@/data";


export default class ImageCard extends Vue {
  @Prop(String) k!: string

  path = images[this.k].value

  update(): void {
    this.path = images[this.k].value
  }

}
</script>

<style scoped>

.image-card {
  max-width: 256px;
  text-align: center;
}

</style>