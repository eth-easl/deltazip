import json
import argparse
import numpy as np
import pandas as pd
import plotly.express as px


def get_provider_name(provider):
    if provider["name"] == "hf":
        return "HuggingFace"
    elif provider["name"] == "fmzip":
        return f"FMZip, bsz={provider['args'].get('batch_size', 1)} <br>strategy={provider['args'].get('placement_strategy','none')}<br>lossy={not provider['args'].get('lossless_only', False)}"

def plot(args):
    print(args)
    with open(args.input, "r") as fp:
        results = json.load(fp)
    plot_data = []
    for item in results:
        provider = item["system"]
        provider = get_provider_name(provider)
        for res in item["results"]:
            plot_data.append(
                {
                    "id": res["response"]["id"],
                    "provider": provider,
                    "time_elapsed": res["time_elapsed"],
                }
            )
    df = pd.DataFrame(plot_data)
    max_time = df["time_elapsed"].max()
    # for each slo_requirement, find how many requests are satisfied
    latency_data = []
    for datum in plot_data:
        latency_data.append({
            "id": datum["id"],
            "latency": datum["time_elapsed"],
            "provider": datum['provider']
        })
    df = pd.DataFrame(latency_data)
    # sort by id
    df = df.sort_values(by=['id', 'provider'])
    fig = px.line(df, x="id", y="latency", color="provider")
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5,
        title_text="Request ID vs. Latency of Different Backends (64 Tokens)",
    )
    fig.update_layout(
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
    )
    fig.update_layout(
        title=dict(font=dict(size=20)),
        legend=dict(font=dict(size=18)),
        legend_title=dict(font=dict(size=18)),
    )
    fig.update_traces(line=dict(width=4))

    fig.update_layout(
        yaxis=dict(
            title_text="Latency", title_font=dict(size=22), tickfont_size=18
        )
    )
    fig.update_layout(
        xaxis=dict(
            title_text="Request ID", title_font=dict(size=22), tickfont_size=18
        )
    )
    fig.write_image(args.output, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    parser.add_argument("--output", type=str, default="artifact/results/latency.png")
    args = parser.parse_args()
    plot(args)
