<template>
  <th scope="row">{{d.name}}</th>
  <td><div class="progress" style="height: 30px;">
    <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar"
         :style="'width: ' + this.percentage(this.d) + '%'"
         :aria-valuenow="this.d.value" aria-valuemin="0" :aria-valuemax="this.d.total">
      {{d.value}} / {{d.total}}
    </div>
  </div></td>
</template>

<script lang="ts">
import { ProgressBarData, progressbars } from "@/data";
import { Prop } from "vue-property-decorator";
import { Vue } from "vue-class-component";


export default class ProgressBarRow extends Vue {
  @Prop(String) k!: string

  d: ProgressBarData = progressbars[this.k]

  percentage(d: ProgressBarData): number {
    return d.value / d.total * 100
  }

  update(): void {
    this.d = progressbars[this.k];
  }

}
</script>
