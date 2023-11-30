import os
import json
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

strategy_mapping = {"none": "None", "addback": "Add-Back", "colocate": "Mixed-Prec"}

tokens = [64, 128, 256, 512]
bits = 4
model_size = "3b"


def get_provider_name(provider):
    if provider["name"] == "hf":
        return "HuggingFace"
    elif provider["name"] == "fmzip":
        return f"FMZip, bsz={provider['args'].get('batch_size', 1)}<br>{strategy_mapping[provider['args'].get('placement_strategy','none')]}<br>lossy={not provider['args'].get('lossless_only', False)}"


def plot(args):
    print(args)
    ar = args.ar
    # if ar does not have a decimal point, convert to int
    if ar % 1 == 0:
        ar = int(ar)
    agg_data = []
    for idx, num_token in enumerate(tokens):
        filename = os.path.join(
            args.input, f"ar_{ar}_{bits}bits_{num_token}tokens.json"
        )
        with open(filename, "r") as fp:
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
            total_jobs = len(item["results"])
            throughput = (
                total_jobs
                / max(item["results"], key=lambda x: x["time_elapsed"])["time_elapsed"]
            )
            agg_data.append(
                {"tokens": num_token, "provider": provider, "throughput": throughput}
            )
    agg_data = pd.DataFrame(agg_data)
    agg_data = agg_data.sort_values(by=["throughput"], ascending=False)
    fig = px.bar(
        agg_data, x="provider", y="throughput", color="provider", facet_col="tokens"
    )
    fig.for_each_annotation(lambda a: a.update(text=f"{a.text.split('=')[-1]} Tokens"))
    fig['layout']['xaxis']['title']['text'] = ""
    fig['layout']['xaxis2']['title']['text'] = ""
    fig['layout']['xaxis3']['title']['text'] = ""
    fig['layout']['xaxis4']['title']['text'] = ""
    fig['layout']['legend']['title']['text'] = ""
    fig.update_layout(
        width=1200,
        height=600,
        title_x=0.5,
        title_y=1,
        title_text=f"Throughput of Different Backends",
        yaxis=dict(
            title_text="Throughput (queries/s)",
            title_font=dict(size=22),
            tickfont_size=18,
        ),
        title=dict(font=dict(size=28)),
        legend=dict(font=dict(size=14)),
        legend_title=dict(font=dict(size=14)),
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
    )
    fig.update_xaxes(showticklabels=False, title_font=dict(size=28), tickfont_size=20)
    fig.update_annotations(
        font_size=24,
        font_color="black",
        font_family="Arial",
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            entrywidth=160,
            yanchor="bottom",
            y=-0.4,
            xanchor="left",
            x=0,
            font=dict(size=24),
        )
    )
    fig.write_image(args.output, scale=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    parser.add_argument("--ar", type=float, default=6)
    parser.add_argument("--output", type=str, default="artifact/results/latency.png")
    args = parser.parse_args()
    plot(args)
