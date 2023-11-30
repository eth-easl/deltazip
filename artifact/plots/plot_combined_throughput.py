import os
import json
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

strategy_mapping = {
    "none": "None",
    "addback": "Add-Back",
    "colocate": "Mixed-Prec"
}

tokens = [64, 128, 256, 512]
bits = 4
model_size = '3b'

def get_provider_name(provider):
    if provider["name"] == "hf":
        return "HuggingFace"
    elif provider["name"] == "fmzip":
        return f"FMZip, bsz={provider['args'].get('batch_size', 1)} <br>{strategy_mapping[provider['args'].get('placement_strategy','none')]}<br>lossy={not provider['args'].get('lossless_only', False)}"

def plot(args):
    print(args)
    ar = args.ar
    # if ar does not have a decimal point, convert to int
    if ar % 1 == 0:
        ar = int(ar)
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        subplot_titles=("64 Tokens", "128 Tokens", "256 Tokens", "512 Tokens"),
        horizontal_spacing=0.015,
        x_title="SLO Requirement (s)",
    )
    for idx, num_token in enumerate(tokens):
        filename = os.path.join(args.input, f"ar_{ar}_{bits}bits_{num_token}tokens.json")
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
        df = pd.DataFrame(plot_data)
        max_time = df["time_elapsed"].max()
        slo_requirements = np.arange(1, max_time, 0.5)
        # for each slo_requirement, find how many requests are satisfied
        slo_data = []
        for slo in slo_requirements:
            for provider in df["provider"].unique():
                provider_df = df[df["provider"] == provider]
                success_rate = (
                    provider_df[provider_df["time_elapsed"] <= slo].shape[0]
                    / provider_df.shape[0]
                )
                slo_data.append(
                    {
                        "slo": slo,
                        "provider": provider,
                        "success_rate": success_rate,
                    }
                )
        df = pd.DataFrame(slo_data)
        fig2 = px.line(df, x="slo", y="success_rate", color="provider")
        fig.add_trace(
            go.Scatter(
                x=fig2["data"][0]["x"],
                y=fig2["data"][0]["y"],
                name=fig2["data"][0]["name"],
                line=dict(color="green", width=4),
                showlegend=True if idx == 0 else False,
            ),
            row=1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=fig2["data"][1]["x"],
                y=fig2["data"][1]["y"],
                name=fig2["data"][1]["name"],
                line=dict(color="red", width=4),
                showlegend=True if idx == 0 else False,
            ),
            row=1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=fig2["data"][2]["x"],
                y=fig2["data"][2]["y"],
                name=fig2["data"][2]["name"],
                line=dict(color="blue", width=4),
                showlegend=True if idx == 0 else False,
            ),
            row=1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=fig2["data"][3]["x"],
                y=fig2["data"][3]["y"],
                name=fig2["data"][3]["name"],
                line=dict(color="purple", width=4),
                showlegend=True if idx == 0 else False,
            ),
            row=1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=fig2["data"][4]["x"],
                y=fig2["data"][4]["y"],
                name=fig2["data"][4]["name"],
                line=dict(color="orange", width=4),
                showlegend=True if idx == 0 else False,
            ),
            row=1,
            col=idx + 1,
        )
    fig.update_layout(
        width=1200,
        height=600,
        title_x=0.5,
        title_text=fr"SLO Attainment of Different Backends",
    )
    fig.update_layout(
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
    )
    fig.update_layout(
        title=dict(font=dict(size=28)),
        legend=dict(font=dict(size=14)),
        legend_title=dict(font=dict(size=14)),
    )
    fig.update_traces(line=dict(width=4))
    fig.update_layout(
        yaxis=dict(
            title_text="Success Rate (%)", title_font=dict(size=22), tickfont_size=18
        )
    )
    fig.update_xaxes(title_font=dict(size=28), tickfont_size=20)
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
            y=-0.5,
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
