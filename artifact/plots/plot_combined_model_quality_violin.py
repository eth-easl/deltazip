import os
import json
import argparse
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

model_mapping = {
    'vicuna': 'Vicuna-7b-v1.5',
    'xwin': 'XWin-7b-v0.1',
}
project_name_mapping = {
    'original': 'Uncompressed',
    'fmzip': 'FiniCompress',
}

def plot(args):
    plot_df = []
    for i, filename in enumerate(
        [x for x in os.listdir(args.input_file) if x.endswith(".jsonl")]
    ):
        with open(os.path.join(args.input_file, filename), "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        
        title = f"Score Distribution 2bits/50% Sparsity Compression Ratio: {data[0]['compression_ratio']}"
        
        for datum in data[1:]:
            for answer in datum["answers"]:
                plot_df.append(
                    {
                        "Model": model_mapping[filename.replace(".jsonl", "")],
                        "source": project_name_mapping[answer["from"]],
                        "Score": answer["score"],
                    }
                )
    df = pd.DataFrame(plot_df)
    fig = px.violin(df, y="Score", x="Model", color="source", box=False)

    fig.update_layout(
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
    )
    fig.update_layout(
        title=dict(font=dict(size=36)),
    )
    fig.update_annotations(
        font_size=24,
        font_color="black",
        font_family="Arial",
    )
    fig.update_xaxes(title_font=dict(size=28), tickfont_size=24)
    fig.update_yaxes(title_font=dict(size=28), tickfont_size=28)
    fig.update_layout(
        width=1200, height=800, title_x=0.5, title_text=title
    )
    # put title a bit up
    fig.update_layout(
        legend=dict(
            title=dict(text="", font=dict(size=28)),
            orientation="h",
            entrywidth=200,
            yanchor="bottom",
            y=-0.2,
            xanchor="left",
            x=0.3,
            font=dict(size=28),
        )
    )
    fig.write_image("artifact/results/images/chat_quality.png", scale=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    args = parser.parse_args()
    plot(args)
