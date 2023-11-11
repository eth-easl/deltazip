import json
import pandas as pd
import plotly.express as px
import argparse


def plot(args):
    print(args)
    with open(args.input, "r") as fp:
        results = json.load(fp)
    plot_data = []
    for item in results:
        backend = item["system"]
        backend_name = backend["name"]
        if backend_name == "hf":
            backend_name = f"{backend_name}<br>bsz={backend['args']['batch_size']}<br>max_models={backend['args']['max_num_models']}"
        elif backend_name == "fmzip":
            backend_name = f"{backend_name}<br>bsz={backend['args']['batch_size']}<br>max_deltas={backend['args']['max_num_deltas']}<br>strategy={backend['args']['placement_strategy']}<br>lossless_only={backend['args']['lossless_only']}"

        min_length = item["gen_configs"]["min_length"]
        for idx, query in enumerate(item["results"]):
            query = query["response"]["response"]
            # total_elapsed = query['total_elapsed']
            tokenize_time = query["measure"]["tokenize_time"]
            loading_time = query["measure"]["loading_time"]
            prepare_time = query["measure"]["prepare_time"]
            inference_time = query["measure"]["inference_time"]
            total_time = tokenize_time + loading_time + prepare_time + inference_time
            plot_data.append(
                {
                    "id": idx,
                    "backend": backend_name,
                    "tokens": min_length,
                    "time_elapsed": tokenize_time,
                    "Breakdown": "Tokenize",
                }
            )
            plot_data.append(
                {
                    "id": idx,
                    "backend": backend_name,
                    "tokens": min_length,
                    "time_elapsed": loading_time,
                    "Breakdown": "Loading",
                }
            )
            plot_data.append(
                {
                    "id": idx,
                    "backend": backend_name,
                    "tokens": min_length,
                    "time_elapsed": prepare_time,
                    "Breakdown": "Prepare",
                }
            )
            plot_data.append(
                {
                    "id": idx,
                    "backend": backend_name,
                    "tokens": min_length,
                    "time_elapsed": inference_time,
                    "Breakdown": "Generation",
                }
            )
    df = pd.DataFrame(plot_data)
    # reverse the order of the dataframe
    # order by name
    df = df.sort_values(by=["backend"], ascending=False)
    fig = px.bar(df, x="backend", y="time_elapsed", color="Breakdown")
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5,
        title_text="Breakdown of Latency (s)<br>Single Query for 3B model; 1x RTX 3090; 2.1 GB/s",
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
    fig.update_yaxes(
        title_text="Time Elapsed", title_font=dict(size=22), tickfont_size=18
    )
    fig.update_xaxes(title_text="Backend", title_font=dict(size=22), tickfont_size=14)
    fig.write_image(args.output, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    parser.add_argument("--output", type=str, default="artifact/plots/latency.png")
    args = parser.parse_args()
    plot(args)
