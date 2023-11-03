import json
import argparse
import pandas as pd
import plotly.express as px


def plot(args):
    print(args)
    with open(args.input, "r") as fp:
        results = json.load(fp)
    plot_data = []
    for item in results:
        provider = item["system"]
        name = "HuggingFace" if provider["name"] == "hf" else "FMZip"
        if name == "FMZip":
            provider = f"{name} bsz={provider['args'].get('batch_size', 1)}<br>{provider['args'].get('placement_strategy', 'none')}<br>Lossy={not provider['args'].get('lossless_only', False)}"
        else:
            provider = "HuggingFace"
        for res in item["results"]:
            plot_data.append(
                {
                    "id": res["response"]["id"],
                    "provider": provider,
                    "time_elapsed": res["time_elapsed"],
                }
            )
    df = pd.DataFrame(plot_data)
    # for each provider, find the maximal time_elapsed
    df = df.groupby(["provider"]).max().reset_index()
    total_jobs = df["id"].max()
    df = df.sort_values(by=["time_elapsed"], ascending=True)
    print(df)
    df['throughput'] = (total_jobs) / df['time_elapsed']
    fig = px.bar(df, x="provider", y="throughput")
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5,
        title_text="Throughput of Different Backends (64 Tokens)",
    )
    fig.update_layout(
        title=dict(font=dict(size=20)),
        legend=dict(font=dict(size=18)),
        legend_title=dict(font=dict(size=18)),
    )
    fig.update_layout(
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
    )
    fig.update_layout(
        yaxis=dict(
            title_text="Throughput (req/s)", title_font=dict(size=22), tickfont_size=18
        )
    )
    fig.update_layout(
        xaxis=dict(title_text="Backend", title_font=dict(size=22), tickfont_size=12)
    )
    fig.write_image(args.output, scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    parser.add_argument("--output", type=str, default="artifact/results/throughput.png")
    args = parser.parse_args()
    plot(args)
