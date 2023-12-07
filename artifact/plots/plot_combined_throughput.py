import os
import json
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from artifact.plots.utils import get_provider_name, get_provider_order, set_plotly_theme, set_font

tokens = [64, 256, 512]
bits = 4
model_size = "3b"
ars = [0.75, 3, 6]

def plot(args):
    fig = make_subplots(
        rows=len(ars),
        cols=len(tokens),
        shared_xaxes=True,
        row_titles=[
            r"$\huge{\lambda=0.75}$",
            r"$\huge{\lambda=3}$",
            r"$\huge{\lambda=6}$",
        ],
        subplot_titles=("64 Tokens", "256 Tokens", "512 Tokens"),
        horizontal_spacing=0.08,
        vertical_spacing=0.04,
        x_title="",
        y_title="Throughput (queries/s)",
        row_heights=[0.333, 0.333, 0.333],
    )
    for ar_id, ar in enumerate(ars):
        for idx, num_token in enumerate(tokens):
            show_legend = True if ar_id == 0 and idx == 0 else False
            agg_data = []
            filename = os.path.join(
                args.input, f"ar_{ar}_{bits}bits_{num_token}tokens.json"
            )
            with open(filename, "r") as fp:
                results = json.load(fp)
            plot_data = []
            for item in results:
                provider = item["system"]
                provider = get_provider_name(provider)
                provider = provider.replace("FiniCompress", "Ours")
                provider = provider.replace("HuggingFace", "HF")
                order = get_provider_order(item["system"])
                for res in item["results"]:
                    plot_data.append(
                        {
                            "id": res["response"]["id"],
                            "provider": provider,
                            "time_elapsed": res["time_elapsed"],
                            "order": order,
                        }
                    )
                total_jobs = len(item["results"])
                last_job = max(
                    item["results"],
                    key=lambda x: x["time_elapsed"] + x["relative_start_at"],
                )
                throughput = total_jobs / (
                    last_job["time_elapsed"] + last_job["relative_start_at"]
                )
                agg_data.append(
                    {
                        "tokens": num_token,
                        "provider": provider,
                        "throughput": throughput,
                        "order": order,
                    }
                )
            agg_data = pd.DataFrame(agg_data)
            agg_data = agg_data.sort_values(by=["order"], ascending=True)
            fig2 = px.bar(
                agg_data,
                x="order",
                y="throughput",
                color="provider",
            )
            fig.add_trace(
                go.Bar(
                    x=fig2["data"][0]["x"],
                    y=fig2["data"][0]["y"],
                    name=fig2["data"][0]["name"],
                    marker=fig2["data"][0]["marker"],
                    showlegend=show_legend,
                ),
                row=ar_id + 1,
                col=idx + 1,
            )
            fig.add_trace(
                go.Bar(
                    x=fig2["data"][1]["x"],
                    y=fig2["data"][1]["y"],
                    name=fig2["data"][1]["name"],
                    marker=fig2["data"][1]["marker"],
                    showlegend=show_legend,
                ),
                row=ar_id + 1,
                col=idx + 1,
            )
            fig.add_trace(
                go.Bar(
                    x=fig2["data"][2]["x"],
                    y=fig2["data"][2]["y"],
                    name=fig2["data"][2]["name"],
                    marker=fig2["data"][2]["marker"],
                    showlegend=show_legend,
                ),
                row=ar_id + 1,
                col=idx + 1,
            )
            fig.add_trace(
                go.Bar(
                    x=fig2["data"][3]["x"],
                    y=fig2["data"][3]["y"],
                    name=fig2["data"][3]["name"],
                    marker=fig2["data"][3]["marker"],
                    showlegend=show_legend,
                ),
                row=ar_id + 1,
                col=idx + 1,
            )
            fig.add_trace(
                go.Bar(
                    x=fig2["data"][4]["x"],
                    y=fig2["data"][4]["y"],
                    name=fig2["data"][4]["name"],
                    marker=fig2["data"][4]["marker"],
                    showlegend=show_legend,
                ),
                row=ar_id + 1,
                col=idx + 1,
            )
    fig["layout"]["annotations"][-1]['xshift'] = -50
    fig.update_layout(
        width=1200,
        height=1200,
        title_x=0.5,
        title_y=1,
        title_text=f"",
        title=dict(font=dict(size=36)),
        legend_title=dict(font=dict(size=16)),
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
    )
    fig.update_xaxes(showticklabels=False, title_font=dict(size=28), tickfont_size=24)
    fig.update_yaxes(showticklabels=True, title_font=dict(size=20), tickfont_size=20)
    fig.update_annotations(
        font=dict(size=30),
        font_color="black",
        font_family="Arial",
    )
    print(fig)
    # enlarge the legend marker
    fig.update_layout(
        legend=dict(
            orientation="h",
            entrywidth=150,
            yanchor="bottom",
            y=-0.18,
            xanchor="left",
            x=0,
            font=dict(size=28),
        )
    )
    fig = set_plotly_theme(fig)
    fig = set_font(fig)
    fig.write_image(args.output, scale=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    parser.add_argument("--output", type=str, default="artifact/results/latency.png")
    args = parser.parse_args()
    plot(args)
