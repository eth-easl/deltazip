import json
import argparse
import numpy as np
import pandas as pd
import plotly.express as px


def get_provider_name(provider):
    if provider["name"] == "hf":
        return "HuggingFace"
    elif provider["name"] == "fmzip":
        return f"FMZip, bsz={provider['args'].get('batch_size', 1)} <br>strategy={provider['args'].get('placement_strategy','none')}<br>lossless={provider['args'].get('lossless_only', False)}"


def plot(args):
    print(args)
    with open(args.input, "r") as fp:
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
    slo_requirements = np.arange(1, 15, 0.1)
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
    print(df)
    fig = px.line(df, x="slo", y="success_rate", color="provider")
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5,
        title_text="SLO Attainment of Different Backends",
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
            title_text="Success Rate (%)", title_font=dict(size=22), tickfont_size=18
        )
    )
    fig.update_layout(
        xaxis=dict(
            title_text="SLO Requirement (s)", title_font=dict(size=22), tickfont_size=18
        )
    )
    fig.write_image(args.output, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    parser.add_argument("--output", type=str, default="artifact/results/latency.png")
    args = parser.parse_args()
    plot(args)
