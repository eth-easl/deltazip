import json
import argparse
import pandas as pd
import plotly.express as px

def plot(args):
    print(args)
    with open(args.input, "r") as fp:
        results = json.load(fp)
    plot_data = []
    for item in results:
        provider = item["system"]
        name = 'HuggingFace' if provider['name'] == 'hf' else 'FMZip'
        provider = f"{name} bsz={provider['args'].get('batch_size', 1)}<br>Lossless Only={provider['args'].get('lossless_only', 'false')}"
        for res in item["results"]:
            plot_data.append(
                {
                    "id": res["response"]["id"],
                    "provider": provider,
                    "time_elapsed": res["time_elapsed"],
                }
            )
    df = pd.DataFrame(plot_data)
    # for each provider, find the maximal time_elapsed
    df = df.groupby(["provider"]).max().reset_index()
    df = df.sort_values(by=["time_elapsed"], ascending=True)
    df['throughput'] = (df['id'] + 1) / df['time_elapsed']
    fig = px.bar(df, x="provider", y="throughput")
    fig.update_layout(
        width=800, height=600, title_x=0.5, title_text="Throughput of Different Backends"
    )
    fig.update_layout(
        title=dict(font=dict(size=20)),
        legend=dict(font=dict(size=18)),
        legend_title=dict(font=dict(size=18)),
    )
    fig.update_layout(
        yaxis=dict(
            title_text="Throughput (req/s)", title_font=dict(size=22), tickfont_size=18
        )
    )
    fig.update_layout(
        xaxis=dict(
            title_text="Backend", title_font=dict(size=22), tickfont_size=14
        )
    )
    fig.write_image("artifact/results/throughput_bar_plot.png", scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    args = parser.parse_args()
    plot(args)
