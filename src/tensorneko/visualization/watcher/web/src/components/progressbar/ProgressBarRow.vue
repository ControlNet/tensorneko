<template>
  <th scope="row">{{ d.name }}</th>
  <td>
    <b-progress striped animated style="height: 30px;" :max="max">
      <b-progress-bar :value="d.value" :variant="variant">
        <span><strong>
          {{d.value}} / {{d.total}}
        </strong></span>
      </b-progress-bar>
    </b-progress>
  </td>
</template>

<script lang="ts">
import { ProgressBarData, progressbars } from "@/data";
import { Prop } from "vue-property-decorator";
import { Vue } from "vue-class-component";


export default class ProgressBarRow extends Vue {
  @Prop(String) k!: string

  d: ProgressBarData = progressbars[this.k]
  max: number = this.d.total
  prev: ProgressBarData = null
  lastModified: number = +new Date()
  variant: Variant = "warning"

  private getType(): Variant {
    const dTime = +new Date() - this.lastModified;

    if (dTime > 10000) {
      return "warning";
    } else if (this.prev !== null) {
      if (this.d.value > this.prev.value) {
        return "success";
      } else if (this.d.value < this.prev.value) {
        return "danger";
      } else {
        return "warning";
      }
    } else {
      return "warning";
    }
  }

  updateVariant(): void {
    this.variant = this.getType()
    setTimeout(this.updateVariant, 10000)
  }

  update(): void {
    this.prev = this.d;
    this.max = this.d.total;
    this.d = progressbars[this.k];
    this.lastModified = +new Date()
  }

  mounted(): void {
    this.updateVariant();
  }

}

type Variant = "success" | "warning" | "danger";

</script>

<style scoped>
div.progress-bar > span {
  color: black;
}
</style>
