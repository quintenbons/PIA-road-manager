import React from 'react';
import Plot from 'react-plotly.js';
import { Data } from "plotly.js";

interface BarChartProps {
    data: Array<{ strategy_name: string, loss_score: number, mutator_id: number }>;
}

const BarChart: React.FC<BarChartProps> = ({ data }) => {
    const trace: Data = {
        x: data.map(item => item.strategy_name),
        y: data.map(item => item.loss_score),
        type: 'bar',
        mode: 'markers',
    };

    return (
        <Plot
            data={[trace]}
            layout={{
                title: 'Loss score per strategy (light traffic)',
                xaxis: { title: 'Strategy' },
                yaxis: { title: 'Score' },
                font: { family: 'Courier New, monospace', size: 18, color: '#7f7f7f' }
            }}
        />
    );
};

export default BarChart;
