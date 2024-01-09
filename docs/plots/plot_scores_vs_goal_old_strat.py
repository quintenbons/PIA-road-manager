# F from torch
import os
import sys
import torch
import polars as pl
import plotly.graph_objects as go

repo_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(os.path.join(repo_path, "src"))

from ai.dataset import get_soft_scores_inverted

lf = pl.scan_csv(sys.argv[1], dtypes={"mutator_id": pl.Utf8, "strategy_name": pl.Utf8, "loss_score": pl.Float64})
lf = lf.with_columns((pl.col("strategy_name") + pl.lit("_") + pl.col("mutator_id")).alias("scheme_name"))
df = lf.collect()

strategy_names = df.select("strategy_name").to_numpy().flatten()

# get soft scores
raw_scores = df.select("loss_score").to_numpy().flatten()
soft_scores = get_soft_scores_inverted(raw_scores)

# add soft scores to df
df = df.with_columns(pl.Series("soft_score", list(soft_scores)))
df = df.with_columns((pl.max("soft_score") - pl.col("soft_score")).alias("inverted_soft_score"))
print(df)

# plot
fig = go.Figure()

df = df.to_pandas()
colors_medium_green = ['#4CAF50', '#4DAE51', '#4CB051', '#49AF4F', '#4CAF52']
colors_vivid_red = ['#F44336', '#F34336', '#F44337', '#F44436', '#F44335']
colors_vivid_blue = ['#2196F3', '#2097F3', '#2196F4', '#2296F3', '#2196F2']

# Iterate over each mutator_id to create a separate bar for each
for mutator_id in df['mutator_id'].unique():
  # Filter the DataFrame for the current mutator_id
  df_filtered = df[df['mutator_id'] == mutator_id]

  # Add a bar for the current mutator_id
  fig.add_trace(go.Bar(
    x=df_filtered['strategy_name'],
    y=df_filtered['loss_score'],
    name=str(mutator_id),  # This will be used in the legend
    marker_color=colors_vivid_blue  # Set red color for scores
  ))
  fig.add_trace(go.Bar(
    x=df_filtered['strategy_name'],
    y=df_filtered['inverted_soft_score'],
    opacity=.4,
    name=str(mutator_id) + " goal",
    yaxis='y2',
    marker_color=colors_vivid_red  # Set green color for softmaxes
  ))

# Update layout (if needed)
fig.update_layout(
  barmode='group',  # Group bars
  height=400,
  width=1000,
  title="Grouped Bar Chart with go.Bar",
  xaxis_title="Strategy Name",
  yaxis_title="Loss Score",
  yaxis=dict(title='Scores'),
  yaxis2=dict(title='Softmaxes', overlaying='y', side='right', showgrid=False),
  legend=dict(x=1.1, y=1),
)

# Show the figure
fig.show()
