import json
import pandas as pd
import plotly.express as px
import argparse


def plot(args):
    print(args)
    with open(args.input, "r") as fp:
        results = json.load(fp)
    plot_data = []
    for item in results:
        provider = item["system"]
        provider = f"{provider['name']}, {provider['args'].get('batch_size', 1)}, {provider['args'].get('placement_strategy','none')}"
        for res in item['results']:
            plot_data.append({
                "id": res['response']['id'],
                "provider": provider,
                "time_elapsed": res['time_elapsed'],
            })
    df = pd.DataFrame(plot_data)
    df = df.sort_values(by=['id'])
    print(df)
    fig = px.line(df, x="id", y="time_elapsed", color="provider")
    fig.update_layout(
        width=800, height=600, title_x=0.5, title_text="Latency (s)"
    )
    fig.write_image("artifact/results/e2e/latency.png", scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    args = parser.parse_args()
    plot(args)
