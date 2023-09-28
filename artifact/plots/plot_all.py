import json
import pandas as pd
import plotly.express as px

with open("artifact/benchmark_results.json", "r") as fp:
    results = json.load(fp)

plot_data = []
for item in results:
    provider = item['provider']
    for result in item['results']:
        plot_data.append({
            'id': result['id'],
            'provider': provider,
            'time_elapsed': result['time_elapsed'],
        })

df = pd.DataFrame(plot_data)

fig = px.histogram(df, x="id", y="time_elapsed",
             color='provider', barmode='group',
             height=400)
# save as file
fig.write_image("artifact/results/latency.png")