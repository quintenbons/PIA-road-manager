#!/usr/bin/env python3
import polars as pl
import plotly.graph_objects as go
import sys

# Read the CSV file
dfs = []
for csvfile in sys.argv[1:]:
    lf = pl.scan_csv(csvfile)
    df = lf.collect()
    dfs.append(df.to_pandas())


# Plot the data
fig = go.Figure()

names=["(10, 64, leaky)", "(4, 64, leaky)", "(1, 128, leaky)", "(4, 64, relu)"]

for df, name in zip(dfs, names):
    fig.add_trace(go.Scatter(
        x=df['epoch'],
        y=df['loss'],
        name=name,
    ))

fig.update_layout(
    title="Loss over epochs on 189k entries (batch size 32)",
    xaxis_title="epoch",
    yaxis_title="loss",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig.update_yaxes(range=[0, None])

fig.show()
# fig.write_html("loss.html")
# fig.write_image("loss.png")
