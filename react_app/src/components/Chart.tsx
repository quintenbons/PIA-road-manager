import React, { useState, useEffect } from 'react';
import BarChart from './BarChart';
import LineChart from './LineChart';
import training_data from '../data/first_training_data.json';
import scores_data from '../data/scores_per_strategy.json';

const chartContainerStyle = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '20px',
};

interface TrainingData {
    epoch: number;
    loss: number;
}

interface ScoreData {
    strategy_name: string;
    loss_score: number;
    mutator_id: number;
}

const Chart = () => {
    const [barChartData, setBarChartData] = useState<ScoreData[]>([]);
    const [lineChartData, setLineChartData] = useState<TrainingData[]>([]);

    useEffect(() => {
        setBarChartData(scores_data as ScoreData[]);
        setLineChartData(training_data as TrainingData[]);
    }, []);

    return (
        <div style={chartContainerStyle}>
            <BarChart data={barChartData} />
            <LineChart data={lineChartData} />
        </div>
    );
};

export default Chart;
