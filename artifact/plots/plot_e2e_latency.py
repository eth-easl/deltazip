import json
import pandas as pd
import plotly.express as px

with open("artifact/benchmark_results.json", "r") as fp:
    results = json.load(fp)

plot_data = []
for item in results:
    provider = item["provider"]
    for result in item["results"]:
        plot_data.append(
            {
                "provider": provider,
                "time_elapsed": result["time_elapsed"],
            }
        )

df = pd.DataFrame(plot_data)
df.sort_values(by=["time_elapsed"], inplace=True)
# allocate id for each row, different providers get different ids
df["id"] = df.groupby("provider").cumcount()
print(df)
# drop index
df.reset_index(drop=True, inplace=True)
# convert ids to string
df["id"] = df["id"].astype(str)
fig = px.histogram(
    df, x="id", y="time_elapsed", color="provider", barmode="group", height=400
)
# set title

# save as file
fig.write_image("artifact/results/latency.png")
