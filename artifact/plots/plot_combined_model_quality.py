import os
import json
import argparse
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot(args):
    print(args)

    fig = make_subplots(rows=1, cols=2)

    for i, filename in enumerate(
        [x for x in os.listdir(args.input_file) if x.endswith(".jsonl")]
    ):
        plot_df = []
        with open(os.path.join(args.input_file, filename), "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        title = f"Score Distribution of {data[0]['model_name']}<br>2bits/50% Sparsity, Compression Ratio: {data[0]['compression_ratio']}"
        for datum in data[1:]:
            for answer in datum["answers"]:
                plot_df.append(
                    {
                        "source": answer["from"],
                        "score": answer["score"],
                    }
                )
        df = pd.DataFrame(plot_df)
        fmzip_score = df[df["source"] == "fmzip"]["score"].to_numpy()
        original_score = df[df["source"] == "original"]["score"].to_numpy()
        avg_fmzip_score = fmzip_score.mean()
        avg_original_score = original_score.mean()
        group_labels = ["FMZip", "Original"]
        score_dist_data = [fmzip_score, original_score]
        fig2 = ff.create_distplot(
            score_dist_data, group_labels, bin_size=1, show_rug=False
        )
        fig.add_trace(
            go.Histogram(fig2["data"][0], marker_color="green"), 
            row=1, col=i + 1
        )

        fig.add_trace(
            go.Histogram(fig2["data"][1], marker_color="red"), 
            row=1, col=i + 1
        )

        fig.add_trace(
            go.Scatter(fig2["data"][2], line=dict(color="green", width=3)),
            row=1, col=i + 1,

        )

        fig.add_trace(
            go.Scatter(fig2["data"][3], line=dict(color="red", width=3)),
            row=1,
            col=i + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=[avg_fmzip_score, avg_fmzip_score],
                y=[0, 0.5],
                mode="lines",
                name="FMZip Average",
                line=dict(color="green", width=3, dash="dash"),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=[avg_original_score, avg_original_score],
                y=[0, 0.5],
                mode="lines",
                name="Original Average",
                line=dict(color="red", width=3, dash="dash"),
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
    )
    fig.update_layout(
        title=dict(font=dict(size=24)),
        legend=dict(font=dict(size=20)),
        legend_title=dict(font=dict(size=20)),
    )
    fig.update_xaxes(title_font=dict(size=22), tickfont_size=20)
    fig.update_layout(width=1200, height=800, title_x=0.5, title_text=title)
    fig.write_image("artifact/results/images/chat_quality.png", scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    args = parser.parse_args()
    plot(args)
