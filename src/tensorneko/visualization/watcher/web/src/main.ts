import { createApp } from 'vue'
import App from './App.vue'
import { ComponentData, ProgressBarData, progressbars, VariableData, variables } from "@/data";


async function initApp(): Promise<void> {
    const response = await fetch(`data.json`);
    const json = await response.json().catch(() => []);
    json.forEach((d: ComponentData) => {
        if (d.type === "Variable") {
            variables[d.name] = d as VariableData;
        } else if (d.type === "ProgressBar") {
            progressbars[d.name] = d as ProgressBarData;
        }
    })
}


initApp().then(() => createApp(App).mount('#app'))
