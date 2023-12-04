import os
import random
import argparse
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

model_mapping = {
    "pythia-2.8b-deduped": "Pythia 2.8b Deduped",
    "open_llama_3b_v2": "OpenLlama 3B V2",
}
project_name_mapping = {"sparsegpt": "SparseGPT", "fmzip": "FiniCompress"}
remove_tasks = ["NI-227"]


def plot(args):
    df = pd.read_csv(args.input)
    base_models = df["base_model"].unique()
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("OpenLlama 3B V2", "Pythia 2.8b Deduped"),
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
        sub_df = sub_df[~sub_df["task_name"].isin(remove_tasks)]
        sub_df["method"] = sub_df["method"].apply(lambda x: project_name_mapping[x])
        fig2 = px.line(
            sub_df,
            y="eval_res",
            x="compression ratio",
            color="task_name",
            line_dash="method",
            markers=True,
        )
        for fig2_data in fig2["data"]:
            fig.add_trace(
                go.Scatter(
                    x=fig2_data["x"],
                    y=fig2_data["y"],
                    name=fig2_data["name"],
                    line=dict(
                        dash="solid" if "FiniCompress" in fig2_data["name"] else "dash",
                        color=fig2_data["line"]["color"],
                    ),
                    mode="lines+markers",
                    showlegend=True if mid == 0 else False,
                ),
                row=1,
                col=mid + 1,
            )
    fig.update_annotations(
        font=dict(size=26),
        font_color="black",
        font_family="Arial",
    )
    # set xaxes to be log scale
    fig.update_traces(line=dict(width=6))

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
    fig.update_xaxes(title=dict(font=dict(size=28)), tickfont_size=24, type="log")
    fig.update_yaxes(title=dict(font=dict(size=28)), tickfont_size=24)
    fig.update_layout(width=1200, height=800, title_x=0.5, title_text=f"Base Model")
    fig.write_image(args.output, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    plot(args)
