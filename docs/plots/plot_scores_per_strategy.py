#!/usr/bin/env python3
import polars as pl
import plotly.express as px
import sys

# Read the CSV file
lf = pl.scan_csv(sys.argv[1], dtypes={"mutator_id": pl.Utf8, "strategy_name": pl.Utf8, "loss_score": pl.Float64})
lf = lf.with_columns((pl.col("strategy_name") + pl.lit("_") + pl.col("mutator_id")).alias("scheme_name"))
df = lf.collect()

# Plot the grouped bars
print(df)
fig = px.bar(df, x="strategy_name", y="loss_score", color='mutator_id', barmode='group', height=400)

fig.update_layout(
    title="Loss score per strategy (light traffic)",
    xaxis_title="strategy",
    yaxis_title="score",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.show()
fig.write_html("score_per_strategy.html")
fig.write_image("score_per_strategy.png")

