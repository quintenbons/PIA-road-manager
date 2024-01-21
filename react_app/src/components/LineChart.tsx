import React from 'react';
import Plot from 'react-plotly.js';
import { Data } from "plotly.js";

interface LineChartProps {
    data: Array<{ epoch: number, loss: number }>;
    minRange: number;
    entryNumber: number
}

const LineChart: React.FC<LineChartProps> = ({ data, minRange = 0, entryNumber }) => {
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
                title: `Loss over epochs on ${entryNumber} entries (batch size 32)`,
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Loss', range: [minRange, Math.max(...data.map(item => item.loss))] },
                font: { family: 'Courier New, monospace', size: 18, color: '#7f7f7f' }
            }}
        />
    );
};

export default LineChart;
