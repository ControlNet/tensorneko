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

export type Variables = { [name: string]: VariableData }
export type Progressbars = { [name: string]: ProgressBarData }
export type Images = { [name: string]: ImageData }

export const variables: Variables = {};
export const progressbars: Progressbars = {};
export const images: Images = {};
