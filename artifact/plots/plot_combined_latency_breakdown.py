import os
import json
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from artifact.plots.utils import get_provider_name, get_provider_order

bits = [2, 4]
tokens = [64, 128]
ar = 0.75
model_size = "3b"
# for latency we only consider a single query, removing batched settings
remove_settings = [
    "FiniCompress<br>bsz=4<br>MMA, Lossy",
    "FiniCompress<br>bsz=8<br>MMA, Lossy",
]
naming_map = {
    "FiniCompress<br>bsz=1<br>MMA, Lossy": "Ours<br>MMA<br>Lossy",
    "FiniCompress<br>bsz=1<br>AS, Lossy": "Ours<br>AS<br>Lossy",
    "FiniCompress<br>bsz=1<br>AS, Lossless": "Ours<br>AS<br>Lossless",
    "HuggingFace": "HF",
}

def plot(args):
    print(args)
    fig = make_subplots(
        rows=len(bits),
        cols=len(tokens),
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=("64 Tokens", "128 Tokens"),
        row_titles=("2 bits", "4 bits"),
        horizontal_spacing=0.015,
        vertical_spacing=0.05,
        x_title="",
        y_title="Time Elapsed (s)",
    )
    for mid, bit in enumerate(bits):
        for idx, num_token in enumerate(tokens):
            show_legend = True if mid == 0 and idx == 0 else False
            filename = os.path.join(
                args.input, f"ar_{ar}_{bit}bits_{num_token}tokens.json"
            )
            with open(filename, "r") as fp:
                results = json.load(fp)
            plot_data = []
            for item in results:
                provider = item["system"]
                num_tokens = item["gen_configs"]["min_length"]
                provider = get_provider_name(provider)
                # use the second query, the first may need warmup
                # ensured the second query is not the same model as the first
                res = item["results"][1]
                tokenize_time = res["response"]["response"]["measure"]["tokenize_time"]
                loading_time = res["response"]["response"]["measure"]["loading_time"]
                prepare_time = res["response"]["response"]["measure"]["prepare_time"]
                inference_time = res["response"]["response"]["measure"][
                    "inference_time"
                ]
                plot_data.append(
                    {
                        "id": res["response"]["id"],
                        "provider": provider,
                        "time_elapsed": tokenize_time,
                        "Breakdown": "Tokenize",
                    }
                )
                plot_data.append(
                    {
                        "id": res["response"]["id"],
                        "provider": provider,
                        "time_elapsed": loading_time,
                        "Breakdown": "Loading",
                    }
                )
                plot_data.append(
                    {
                        "id": res["response"]["id"],
                        "provider": provider,
                        "time_elapsed": prepare_time,
                        "Breakdown": "Prepare",
                    }
                )
                plot_data.append(
                    {
                        "id": res["response"]["id"],
                        "provider": provider,
                        "time_elapsed": inference_time,
                        "Breakdown": "Inference",
                    }
                )
            df = pd.DataFrame(plot_data)
            df = df[~df["provider"].isin(remove_settings)]
            df["provider"] = df["provider"].apply(lambda x: naming_map[x])
            fig2 = px.bar(df, x="provider", y="time_elapsed", color="Breakdown")
            for fig2_data in fig2["data"]:
                fig.add_trace(
                    go.Bar(
                        x=fig2_data["x"],
                        y=fig2_data["y"],
                        name=fig2_data["name"],
                        marker=fig2_data["marker"],
                        showlegend=show_legend,
                    ),
                    row=mid + 1,
                    col=idx + 1,
                )
    fig.update_layout(barmode="stack")
    fig.update_layout(
        width=1200,
        height=800,
        title_x=0.5,
        title_text=f"Latency Breakdown of Different Backends",
    )
    fig.update_annotations(
        font=dict(size=28),
        font_color="black",
        font_family="Arial",
    )
    fig.update_layout(
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        title_font_size=28,
        legend_title_font_color="black",
    )
    fig.update_layout(yaxis=dict(title_font=dict(size=22), tickfont_size=18))
    fig.update_annotations(
        font_size=28,
        font_color="black",
        font_family="Arial",
    )
    fig.update_xaxes(title_font=dict(size=24), tickfont_size=24)
    fig.update_layout(
        legend=dict(
            orientation="h",
            entrywidth=160,
            yanchor="bottom",
            y=-0.3,
            xanchor="left",
            x=0.1,
            font=dict(size=28),
        )
    )
    fig.write_image(args.output, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    parser.add_argument("--output", type=str, default="artifact/results/latency.png")
    args = parser.parse_args()
    plot(args)
