import React from 'react';
import Plot from 'react-plotly.js';
import { Data } from "plotly.js";

interface LineChartProps {
    data: Array<{ epoch: number, loss: number }>;
}

const LineChart: React.FC<LineChartProps> = ({ data }) => {
    const trace: Data = {
        x: data.map(item => item.epoch),
        y: data.map(item => item.loss),
        type: 'scatter',
        mode: 'lines+markers',
    };
    return (
        <Plot
            data={[trace]}
            layout={{
                title: 'Loss over epochs on 400 entries (batch size 32)',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Loss', range: [0, Math.max(...data.map(item => item.loss))] },
                font: { family: 'Courier New, monospace', size: 18, color: '#7f7f7f' }
            }}
        />
    );
};

export default LineChart;
