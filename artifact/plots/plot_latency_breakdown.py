import os
import json
import argparse
import pandas as pd
import plotly.express as px

strategy_mapping = {"none": "None", "addback": "Add-Back", "colocate": "Mixed-Prec"}


def get_provider_name(provider):
    if provider["name"] == "hf":
        return "HuggingFace"
    elif provider["name"] == "fmzip":
        return f"FMZip, bsz={provider['args'].get('batch_size', 1)} <br>strategy={strategy_mapping[provider['args'].get('placement_strategy','none')]}<br>lossy={not provider['args'].get('lossless_only', False)}"


def plot(args):
    print(args)
    with open(args.input, "r") as fp:
        results = json.load(fp)
    plot_data = []
    for item in results:
        provider = get_provider_name(item["system"])
        num_tokens = item["gen_configs"]["min_length"]
        res = item["results"][0]
        tokenize_time = res["response"]["response"]["measure"]["tokenize_time"]
        loading_time = res["response"]["response"]["measure"]["loading_time"]
        prepare_time = res["response"]["response"]["measure"]["prepare_time"]
        inference_time = res["response"]["response"]["measure"]["inference_time"]
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
    fig = px.bar(df, x="provider", y="time_elapsed", color="Breakdown")
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5,
        title_text=f"Latency Breakdown, {num_tokens} tokens<br>{args.gpu_spec}, {args.disk_bw}",
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

    fig.update_layout(
        yaxis=dict(
            title_text=f"Time Elapsed", title_font=dict(size=22), tickfont_size=18
        )
    )
    fig.update_layout(
        xaxis=dict(title_text="Backend", title_font=dict(size=22), tickfont_size=12)
    )
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    fig.write_image(args.output, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    parser.add_argument("--output", type=str, default="artifact/results/latency.png")
    parser.add_argument("--token-count", type=int, default=64)
    parser.add_argument("--gpu-spec", type=str, default="RTX 3090")
    parser.add_argument("--disk-bw", type=str, default="2.1 GB/s")
    args = parser.parse_args()
    plot(args)
