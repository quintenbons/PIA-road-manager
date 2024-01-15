#!/usr/bin/env python3
# F from torch
import os
import sys
import torch
import polars as pl
import plotly.graph_objects as go

repo_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(os.path.join(repo_path, "src"))

from ai.dataset import get_soft_scores

lf = pl.scan_csv(sys.argv[1], dtypes={"mutator_id": pl.Utf8, "strategy_name": pl.Utf8, "loss_score": pl.Float64})
lf = lf.with_columns((pl.col("strategy_name") + pl.lit("_") + pl.col("mutator_id")).alias("scheme_name"))
df = lf.collect()

strategy_names = df.select("strategy_name").to_numpy().flatten()

df = df.with_columns((pl.max("target_score") - pl.col("target_score")).alias("inverted_target_score"))
df = df.with_columns((pl.max("output_score") - pl.col("output_score")).alias("inverted_output_score"))

# Careful: this is currently hardcoded. Source: bench_ai.py
mean_score = 2*df['loss_score'].min()

# plot
fig = go.Figure()

df = df.to_pandas()
colors_background_blue = '#8cb9ca'
colors_foreground_blue = '#135589'
colors_foreground_green = '#165711'

fig.add_hline(y=mean_score, line_width=3, line_dash="dash", line_color="red")

# Iterate over each mutator_id to create a separate bar for each
for mutator_id in df['mutator_id'].unique():
  # Filter the DataFrame for the current mutator_id
  df_filtered = df[df['mutator_id'] == mutator_id]

  # Add a bar for the current mutator_id
  fig.add_trace(go.Bar(
    x=df_filtered['strategy_name'],
    y=df_filtered['loss_score'],
    name="score",  # This will be used in the legend
    marker_color=colors_background_blue,  # Set green color for softmaxes
    legendgroup="scores",
    showlegend=mutator_id == df['mutator_id'].unique()[0],
  ))
  fig.add_trace(go.Bar(
    x=df_filtered['strategy_name'],
    y=df_filtered['inverted_target_score'],
    name="goal",
    yaxis='y2',
    marker_color=colors_foreground_blue , # Set red color for scores
    legendgroup="goal",
    showlegend=mutator_id == df['mutator_id'].unique()[0],
    width=0.04,
  ))
  fig.add_trace(go.Bar(
    x=df_filtered['strategy_name'],
    y=df_filtered['inverted_output_score'],
    name="output",
    yaxis='y2',
    marker_color=colors_foreground_green, # Set red color for scores
    legendgroup="goal",
    showlegend=mutator_id == df['mutator_id'].unique()[0],
    width=0.04,
  ))

# Update layout (if needed)
fig.update_layout(
  barmode='group',  # Group bars
  height=500,
  title="Compare model to goal",
  xaxis_title="Strategy Name",
  yaxis_title="Loss Score",
  yaxis=dict(title='Scores'),
  yaxis2=dict(title='Softmaxes', overlaying='y', side='right', showgrid=False),
  legend=dict(x=1.1, y=1),
  bargroupgap=0.05,
)

# Show the figure
fig.show()
