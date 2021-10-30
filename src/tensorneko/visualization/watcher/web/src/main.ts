import { createApp } from 'vue'
import BootstrapVue3 from "bootstrap-vue-3";
import App from './App.vue'
import { App as AppRuntime } from "vue"
import "../node_modules/bootstrap/dist/css/bootstrap.min.css";


async function initApp(): Promise<AppRuntime<Element>> {
    return createApp(App)
}


initApp().then(app => {
    app.use(BootstrapVue3);
    app.mount('#app');
})
