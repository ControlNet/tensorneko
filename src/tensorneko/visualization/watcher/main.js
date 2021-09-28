const variables = {};
const progressbars = {};

async function initApp() {
    const response = await fetch("data.json");
    const json = await response.json().catch(() => []);
    json.forEach(d => {
        if (d.type === "Variable") {
            variables[d.name] = d;
        } else if (d.type === "ProgressBar") {
            progressbars[d.name] = d;
        }
    })

    // app
    const app = Vue.createApp({
        data() {
            return { variables, progressbars, updateDuration: 1000 }
        },

        methods: {
            update(cache = "no-cache") {
                fetch("data.json", {cache: cache})
                    .then(response => response.json())
                    .then(json => {
                        const changes = [];

                        function updateArray(d, array) {
                            const prev = array[d.name];
                            if (prev === undefined) {
                                array[d.name] = d;
                            } else if (prev.value !== d.value) {
                                array[d.name] = d;
                                changes.push({ type: d.type, name: d.name })
                            }
                        }

                        json.forEach(d => {
                            if (d.type === "Variable") {
                                updateArray(d, variables);
                            } else if (d.type === "ProgressBar") {
                                updateArray(d, progressbars);
                            }
                        });

                        changes.forEach(({type, name}) => {
                            if (type === "ProgressBar") {
                                this.$refs.progressbar.$refs[name].update();
                            } else if (type === "Variable") {
                                this.$refs.variable.$refs[name].update();
                            }
                        });

                        setTimeout(this.update, Math.max(100, this.updateDuration));
                    })
                    .catch(() => {
                        setTimeout(() => this.update("reload"), Math.max(100, this.updateDuration))
                    });
            }
        },
        mounted() {
            this.update();
        },
        template: `
            <div class="row mb-3">
                <label class="col-sm-2 col-form-label" for="exampleFormControlInput1" style="width: 200px">Update Rate (ms)</label>
                <div class="col-sm-10" style="width: 200px;"><input type="number" class="form-control" placeholder="1000" v-model="updateDuration"></div>
            </div>
            <progressbar-table v-if="Object.keys(progressbars).length > 0" ref="progressbar"/>
            <variable-table v-if="Object.keys(variables).length > 0" ref="variable"/>
        `
    })

    // progressbar table
    app.component("progressbar-table", {
        data() {
            return { progressbars }
        },
        template: `
            <h6 class="display-6">ProgressBars</h6>
            <table class="table table-striped table-hover">
                <thead><tr>
                    <th scope="col" style="width: 200px;">Name</th>
                    <th scope="col"></th>
                </tr></thead>
                <tbody><tr v-for="k of Object.keys(progressbars)" class="table-secondary">
                    <progressbar-row :k="k" :ref="k"/>
                </tr></tbody>
            </table>
        `
    })

    app.component("progressbar-row", {
        props: ["k"],
        data() {
            return {
                d: progressbars[this.k]
            }
        },
        methods: {
            percentage(d) {
                return d.value / d.total * 100
            },

            update() {
                this.d = progressbars[this.k];
            }
        },
        template: `
            <th scope="row">{{d.name}}</th>
            <td><div class="progress" style="height: 30px;">
                <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar"
                    :style="'width: ' + percentage(d) + '%'"
                    :aria-valuenow="d.value" aria-valuemin="0" :aria-valuemax="d.total">
                    {{d.value}} / {{d.total}}
                </div>
            </div></td>
        `
    })

    // variable table
    app.component("variable-table", {
        data() {
            return { variables }
        },
        template: `
            <h6 class="display-6">Variables</h6>
            <table class="table table-striped table-hover">
                <thead><tr>
                    <th scope="col" style="width: 200px;">Name</th>
                    <th scope="col">Value</th>
                </tr></thead>
                <tbody><tr v-for="k of Object.keys(variables)">
                    <variable-row :k="k" :ref="k" />
                </tr></tbody>
            </table>
        `
    });

    app.component("variable-row", {
        props: ["k"],
        data() {
            return {
                d: variables[this.k]
            }
        },
        methods: {
            update() {
                this.d = variables[this.k];
            }
        },

        template: `
            <th scope="row">{{d.name}}</th>
            <td>{{d.value}}</td>
        `
    });

    return app;
}

initApp().then(app => app.mount("#app"));