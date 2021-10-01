<template>
  <h2 class="display-2">Tensorneko Web Watcher</h2>
  <div class="row mb-3">
    <label class="col-sm-2 col-form-label" style="width: 200px">Update Rate (ms)</label>
    <div class="col-sm-10" style="width: 200px;"><input type="number" class="form-control" placeholder="1000"
                                                        v-model="updateDuration"></div>
  </div>
  <ProgressBarTable v-if="Object.keys(this.progressbars).length > 0" ref="progressbar"/>
  <VariableTable v-if="Object.keys(this.variables).length > 0" ref="variable"/>
  <ImageSection v-if="Object.keys(this.images).length > 0" ref="image"/>
</template>

<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import { ComponentData, variables, progressbars, images } from "@/data";
import ProgressBarTable from "@/components/ProgressBarTable.vue";
import VariableTable from "@/components/VariableTable.vue";
import ImageSection from "@/components/ImageSection.vue";

type RequestCache = "default" | "force-cache" | "no-cache" | "no-store" | "only-if-cached" | "reload";


@Options({
  components: {
    ProgressBarTable,
    VariableTable,
    ImageSection
  },
})
export default class App extends Vue {
  readonly variables = variables
  readonly progressbars = progressbars
  readonly images = images
  updateDuration = 5000

  $refs!: {
    progressbar: ProgressBarTable,
    variable: VariableTable,
    image: ImageSection
  }

  update(cache: RequestCache = "no-cache"): void {
    fetch("/data.json", {cache: cache})
        .then(response => response.json())
        .then(json => {
          const changes: Array<{type: string, name: string}> = [];

          function updateArray(d: ComponentData, array: {[name: string]: ComponentData}) {
            const prev = array[d.name];
            if (prev === undefined) {
              array[d.name] = d;
            } else if (prev.value !== d.value) {
              array[d.name] = d;
              changes.push({type: d.type, name: d.name});
            }
          }

          json.forEach((d: ComponentData) => {
            if (d.type === "Variable") {
              updateArray(d, this.variables);
            } else if (d.type === "ProgressBar") {
              updateArray(d, this.progressbars);
            } else if (d.type === "Image") {
              updateArray(d, this.images);
            }
          });

          changes.forEach(({type, name}) => {
            if (type === "ProgressBar") {
              this.$refs.progressbar.$refs[name].update();
            } else if (type === "Variable") {
              this.$refs.variable.$refs[name].update();
            } else if (type === "Image") {
              this.$refs.image.$refs[name].update();
            }
          });

          setTimeout(this.update, Math.max(100, this.updateDuration));
        })
        .catch(() => {
          setTimeout(() => this.update("reload"), Math.max(100, this.updateDuration))
        });
  }

  mounted(): void {
    this.update();
  }
}
</script>

<style>
body {
  margin: 10px !important;
}
</style>
