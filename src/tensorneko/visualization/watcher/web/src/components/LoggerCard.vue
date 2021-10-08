<template>
  <b-card>
    <b-card :title="k">
      <b-card-text v-for="text of content" :key="text">{{text}}</b-card-text>
    </b-card>
  </b-card>
</template>

<script lang="ts">
import { Vue } from "vue-class-component";
import { Prop } from "vue-property-decorator";
import { logs } from "@/data.ts";

export default class LoggerCard extends Vue {
  @Prop(String) k!: string

  content: Array<string> = logs[this.k].value

  update(): void {
    this.content = logs[this.k].value
  }
}
</script>

<style scoped>

</style>