import { reactive } from "vue";

export declare abstract class ComponentData {
    name: string
    type: string
    value: unknown
}

export declare class VariableData extends ComponentData {
    type: "Variable"
    value: string
}

export declare class ProgressBarData extends ComponentData {
    type: "ProgressBar"
    value: number
    total: number
}

export declare class ImageData extends ComponentData {
    type: "Image"
    value: string
}

export declare class LoggerData extends ComponentData {
    type: "Logger"
    value: Array<string>
}

export declare class LineChartData extends ComponentData {
    type: "LineChart"
    x_label: string
    y_label: string
    value: Array<{
        x: number
        y: number
        label: string
    }>
}

export type Variables = { [name: string]: VariableData }
export type Progressbars = { [name: string]: ProgressBarData }
export type Images = { [name: string]: ImageData }
export type Logs = { [name: string]: LoggerData }
export type LineCharts = { [name: string]: LineChartData }

export const variables: Variables = reactive({});
export const progressbars: Progressbars = reactive({});
export const images: Images = reactive({});
export const logs: Logs = reactive({});
export const lineCharts: LineCharts = reactive({});
