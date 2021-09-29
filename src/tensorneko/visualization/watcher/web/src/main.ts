import { createApp } from 'vue'
import BootstrapVue3 from "bootstrap-vue-3";
import App from './App.vue'
import { App as AppRuntime } from "vue"
import { ComponentData, ProgressBarData, progressbars, VariableData, variables } from "@/data";
import "../node_modules/bootstrap/dist/css/bootstrap.min.css";


async function initApp(): Promise<AppRuntime<Element>> {
    const response = await fetch(`data.json`);
    const json = await response.json().catch(() => []);
    json.forEach((d: ComponentData) => {
        if (d.type === "Variable") {
            variables[d.name] = d as VariableData;
        } else if (d.type === "ProgressBar") {
            progressbars[d.name] = d as ProgressBarData;
        }
    })
    return createApp(App)
}


initApp().then(app => {
    app.use(BootstrapVue3);
    app.mount('#app');
})
