<script setup lang="ts">
import { type ComponentData, variables, progressbars, images, logs, lineCharts } from "@/data";
import ProgressBarTable from "@/components/progressbar/ProgressBarTable.vue";
import VariableTable from "@/components/variable/VariableTable.vue";
import ImageSection from "@/components/image/ImageSection.vue";
import LoggerSection from "@/components/logger/LoggerSection.vue";
import PlotSection from "@/components/plot/PlotSection.vue";
import { onMounted, type Ref, ref } from "vue";
import _ from "lodash";

type RequestCache = "default" | "force-cache" | "no-cache" | "no-store" | "only-if-cached" | "reload";
const updateDuration = ref(5000)
const view: Ref<string> = ref("")
const progressbar = ref<InstanceType<typeof ProgressBarTable> | null>(null)
const variable = ref<InstanceType<typeof VariableTable> | null>(null)
const image = ref<InstanceType<typeof ImageSection> | null>(null)
const logger = ref<InstanceType<typeof LoggerSection> | null>(null)
const plot = ref<InstanceType<typeof PlotSection> | null>(null)

function update(cache: RequestCache = "no-cache"): void {
  fetch("/data.json", {cache: cache})
      .then(response => response.json())
      .then(json => {
        const changes: Array<{ type: string, name: string }> = [];

        function updateArray(d: ComponentData, array: { [name: string]: ComponentData }) {
          const prev: ComponentData | undefined = array[d.name];
          if (prev === undefined) {
            array[d.name] = d;
          } else if (!_.isEqual(prev?.value, d.value)) {
            array[d.name] = d;
            changes.push({type: d.type, name: d.name});
          }
        }

        view.value = json.view;

        json.data.forEach((d: ComponentData) => {
          if (d.type === "Variable") {
            updateArray(d, variables);
          } else if (d.type === "ProgressBar") {
            updateArray(d, progressbars);
          } else if (d.type === "Image") {
            updateArray(d, images);
          } else if (d.type === "Logger") {
            updateArray(d, logs)
          } else if (d.type === "LineChart") {
            updateArray(d, lineCharts)
          }
        });

        changes.forEach(({type, name}) => {
          if (type == "ProgressBar") {
            progressbar.value?.subs.get(name)?.update();
          } else if (type == "Variable") {
            variable.value?.subs.get(name)?.update();
          } else if (type == "Image") {
            image.value?.subs.get(name)?.update();
          } else if (type == "Logger") {
            logger.value?.subs.get(name)?.update();
          } else if (type == "LineChart") {
            plot.value?.subs.get(name)?.update();
          }
        });

        setTimeout(update, Math.max(100, updateDuration.value));
      })
      .catch(() => {
        setTimeout(() => update("reload"), Math.max(100, updateDuration.value))
      });
}

onMounted(() => update())

function nPlots() {
  return Object.keys(lineCharts).length
}
</script>

<template>
  <h2 class="display-2">Tensorneko Web Watcher::{{ view }}</h2>
  <div class="row mb-3">
    <label class="col-sm-2 col-form-label" style="width: 200px">Update Rate (ms)</label>
    <div class="col-sm-10" style="width: 200px;"><input type="number" class="form-control" placeholder="1000"
                                                        v-model="updateDuration"></div>
  </div>
  <ProgressBarTable v-if="Object.keys(progressbars).length > 0" ref="progressbar"/>
  <VariableTable v-if="Object.keys(variables).length > 0" ref="variable"/>
  <PlotSection v-if="nPlots() > 0" ref="plot"/>
  <ImageSection v-if="Object.keys(images).length > 0" ref="image"/>
  <LoggerSection v-if="Object.keys(logs).length > 0" ref="logger"/>
</template>

