import os
import random
import argparse
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from artifact.plots.utils import set_plotly_theme, set_font

model_mapping = {
    "pythia-2.8b-deduped": "Pythia 2.8b Deduped",
    "open_llama_3b_v2": "OpenLlama 3B V2",
}
project_name_mapping = {"sparsegpt": "SparseGPT", "fmzip": "DeltaZip"}
poi_tasks = ["NI-151", "NI-380", "NI-1308"]

raw_symbols = SymbolValidator().values


def plot(args):
    df = pd.read_csv(args.input)
    base_models = df["base_model"].unique()
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("OpenLlama-3B-V2", "Pythia-2.8b-deduped"),
        horizontal_spacing=0.015,
        x_title="Compression Ratio (log-scale)",
        y_title="Downstream Accuracy",
    )
    df = df.sort_values(by=["base_model"], ascending=True)
    for mid, model in enumerate(base_models):
        sub_df = df[df["base_model"] == model]
        sub_df = sub_df.sort_values(by=["compression ratio"])
        sub_df["task_name"] = sub_df["task"].apply(
            lambda x: x.split("_")[0].replace("task", "NI-")
        )
        sub_df = sub_df[sub_df["task_name"].isin(poi_tasks)]
        sub_df["method"] = sub_df["method"].apply(lambda x: project_name_mapping[x])
        symbols = ["square", "circle", "diamond", "cross"]
        fig2 = px.line(
            sub_df,
            y="eval_res",
            x="compression ratio",
            symbol="task_name",
            color="task_name",
            symbol_sequence=symbols,
            line_dash="method",
            markers=True,
        )
        for fig2_data in fig2["data"]:
            fig.add_trace(
                go.Scatter(
                    x=fig2_data["x"],
                    y=fig2_data["y"],
                    name=fig2_data["name"].split(",")[0],
                    line=dict(
                        dash="solid" if "FiniCompress" in fig2_data["name"] else "dash",
                        color=fig2_data["line"]["color"],
                    ),
                    marker=fig2_data["marker"],
                    mode="lines+markers",
                    showlegend=True
                    if mid == 0 and "FiniCompress" in fig2_data["name"]
                    else False,
                ),
                row=1,
                col=mid + 1,
            )
    fig.update_annotations(
        font=dict(size=32),
        font_color="black",
        font_family="Arial",
    )
    fig["layout"]["annotations"][2]["yshift"] = -50
    fig["layout"]["annotations"][3]["xshift"] = -50

    # set xaxes to be log scale
    fig.update_traces(line=dict(width=5))
    fig.update_traces(marker={"size": 15})
    fig.update_layout(
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
    )
    fig.update_layout(
        title=dict(font=dict(size=36)),
        legend=dict(font=dict(size=24)),
        legend_title=dict(font=dict(size=24)),
    )
    fig.update_xaxes(type="log")
    fig.update_layout(width=1250, height=800, title_x=0.5, title_text=f"Base Model")
    # set background color to white, with grid lines
    fig.update_layout(
        legend=dict(
            orientation="h",
            entrywidth=120,
            yanchor="bottom",
            y=-0.25,
            xanchor="left",
            x=0.2,
            font=dict(size=24),
        ),
    )
    fig = set_font(fig)
    fig = set_plotly_theme(fig)
    fig.write_image(args.output, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    plot(args)
