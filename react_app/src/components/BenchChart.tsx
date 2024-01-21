import React from 'react';
import Plot from 'react-plotly.js';
import { Data } from "plotly.js";

interface DataType {
    "strategy_id": number;
    "strategy_name": string;
    "mutator_id": string;
    "loss_score": number;
    "target_score": number;
    "output_score": number;
    "scheme_name": string;
    "inverted_target_score": number;
    "inverted_output_score": number;
}

interface BenchChartProps {
    data: Array<DataType>;
}

const BenchChart: React.FC<BenchChartProps> = ({ data }) => {
    const strategy_names = Array.from(new Set(data.map(item => item.strategy_name)));
    const mutator_ids = Array.from(new Set(data.map(item => item.mutator_id)));

    const colScore = '#8cb9ca'
    const colGoal = '#135589'
    const colOutput = '#165711'

    const minScore = Math.min(...data.map(item => item.loss_score))
    const meanScore = 2 * minScore; // Empiric

    let traces: Data[] = []

    for (const mutator_id of mutator_ids) {
        const scoreTrace: Data = {
            x: strategy_names,
            y: data.filter(item => item.mutator_id === mutator_id).map(item => item.loss_score),
            name: "Score",
            type: 'bar',
            mode: 'markers',
            marker: {
                color: colScore
            },
            yaxis: 'y1',
            legendgroup: "hard",
            showlegend: mutator_id === "0",
        };
        const goalTrace: Data = {
            x: strategy_names,
            y: data.filter(item => item.mutator_id === mutator_id).map(item => item.inverted_target_score),
            name: "Goal",
            type: 'bar',
            mode: 'markers',
            marker: {
                color: colGoal
            },
            yaxis: 'y2',
            legendgroup: "soft",
            showlegend: mutator_id === "0",
            width: 0.04,
        };
        const output: Data = {
            x: strategy_names,
            y: data.filter(item => item.mutator_id === mutator_id).map(item => item.inverted_output_score),
            name: "Output",
            type: 'bar',
            mode: 'markers',
            marker: {
                color: colOutput
            },
            yaxis: 'y2',
            legendgroup: "soft",
            showlegend: mutator_id === "0",
            width: 0.04,
        };

        traces.push(scoreTrace);
        traces.push(goalTrace);
        traces.push(output);
    }

    return (
        <Plot
            data={traces}
            layout={{
                title: "Compare model to goal",
                height: 600,
                xaxis: { title: 'Strategy Name' },
                yaxis: { title: 'Loss Scores' },
                yaxis2: {title: "Softmax Scores", overlaying: 'y', side: 'right', range: [0, 0.2] },
                font: { family: 'Courier New, monospace', size: 18, color: '#7f7f7f' },
                legend: { x: 1.1, y: 1 },
                barmode: "group",
                bargroupgap: 0.05,
                shapes: [
                    {
                        type: 'line',
                        xref: 'paper',
                        x0: 0,
                        x1: 1,
                        y0: meanScore,
                        y1: meanScore,
                        line: {
                            color: 'rgb(255, 0, 0)',
                            width: 4,
                            dash: 'dot',
                        },
                    }
                ],
            }}
        />
    );
};

export default BenchChart;
