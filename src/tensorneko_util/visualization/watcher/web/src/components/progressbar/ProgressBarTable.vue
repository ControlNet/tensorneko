<template>
  <h6 class="display-6">ProgressBars</h6>
  <table class="table table-striped table-hover">
    <thead>
    <tr>
      <th scope="col" style="width: 200px;">Name</th>
      <th scope="col"/>
    </tr>
    </thead>
    <tbody>
    <tr v-for="k of Object.keys(progressbars)" :key="k" class="table-secondary">
      <ProgressBarRow :k="k" :ref="k"/>
    </tr>
    </tbody>
  </table>
</template>

<script lang="ts">
import { Options, Vue } from "vue-class-component";

import { progressbars } from "@/data";
import ProgressBarRow from "@/components/progressbar/ProgressBarRow.vue";


@Options({
  components: { ProgressBarRow }
})
export default class ProgressBarTable extends Vue {
  progressbars = progressbars;

  $refs!: Record<string, ProgressBarRow>
}
</script>
