import React from 'react';
import Plot from 'react-plotly.js';
import { Data } from "plotly.js";

interface BarChartProps {
    data: Array<{ strategy_name: string, loss_score: number, mutator_id: number }>;
}

const BarChart: React.FC<BarChartProps> = ({ data }) => {
    const strategy_names = Array.from(new Set(data.map(item => item.strategy_name)));
    const mutator_ids = Array.from(new Set(data.map(item => item.mutator_id)));

    let traces: Data[] = []

    for (const mutator_id of mutator_ids) {
        const trace: Data = {
            x: strategy_names,
            y: data.filter(item => item.mutator_id === mutator_id).map(item => item.loss_score),
            type: 'bar',
            mode: 'markers',
        };
        traces.push(trace);
    }

    return (
        <Plot
            data={traces}
            layout={{
                title: 'Loss score per strategy (light traffic)',
                xaxis: { title: 'Strategy' },
                yaxis: { title: 'Score' },
                font: { family: 'Courier New, monospace', size: 18, color: '#7f7f7f' },
                barmode: "group",
            }}
        />
    );
};

export default BarChart;
