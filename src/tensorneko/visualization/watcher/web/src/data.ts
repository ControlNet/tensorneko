export declare class ComponentData {
    name: string
    type: string
    value: any
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

export const variables: { [name: string]: VariableData } = {};
export const progressbars: { [name: string]: ProgressBarData } = {};
