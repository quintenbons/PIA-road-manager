#!/usr/bin/env python3
# csv output: key_size,gen_time
import polars as pl
import plotly.express as px
import sys

# Read the CSV file
lf = pl.scan_csv(sys.argv[1])
df = lf.collect()

# Plot the data
fig = px.line(df, x="epoch", y="loss")
fig.update_layout(
    title="Loss over epochs on 400 entries (batch size 32)",
    xaxis_title="epoch",
    yaxis_title="loss",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig.show()
fig.write_html("loss.html")
