import os
import json
import argparse
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot(args):
    print(args)
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("vicuna-7b-v1.5", "xwin-7b-v0.1"),
        horizontal_spacing=0.015,
        x_title="Score",
        y_title="Percentage of Responses",
    )
    plot_df = []
    for i, filename in enumerate(
        [x for x in os.listdir(args.input_file) if x.endswith(".jsonl")]
    ):
        with open(os.path.join(args.input_file, filename), "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        title = f"Score Distribution 2bits/50% Sparsity<br>Compression Ratio: {data[0]['compression_ratio']}"
        for datum in data[1:]:
            for answer in datum["answers"]:
                plot_df.append(
                    {
                        "model": filename.replace(".jsonl", ""),
                        "source": answer["from"],
                        "score": answer["score"],
                    }
                )
    df = pd.DataFrame(plot_df)
    print(df)
    fig = px.violin(df, y="score", x="model", color="source", box=True, points="all",
          hover_data=df.columns)

    fig.update_layout(
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
    )
    fig.update_layout(
        title=dict(font=dict(size=28)),
        legend=dict(font=dict(size=20)),
        legend_title=dict(font=dict(size=20)),
    )
    fig.update_annotations(
        font_size=24,
        font_color="black",
        font_family="Arial",
    )
    fig.update_xaxes(title_font=dict(size=24), tickfont_size=24)
    fig.update_layout(
        width=1200, height=800, title_x=0.5, title_text=title, title_y=0.96
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            entrywidth=200,
            yanchor="bottom",
            y=-0.2,
            xanchor="left",
            x=0,
            font=dict(size=24),
        )
    )
    fig.write_image("artifact/results/images/chat_quality.png", scale=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    args = parser.parse_args()
    plot(args)
