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
        provider = item["backend"]
        provider = f"{provider['name']}, {provider['args'].get('batch_size', 1)}, {provider['args'].get('model_parallel_strategy', 'none')}"
        for res in item["results"]:
            tokenize_time = res["response"]["response"]["measure"]["tokenize_time"]
            loading_time = res["response"]["response"]["measure"]["loading_time"]
            prepare_time = res["response"]["response"]["measure"]["prepare_time"]
            inference_time = res["response"]["response"]["measure"]["inference_time"]
            waiting_time = res["time_elapsed"] - (
                tokenize_time + loading_time + prepare_time + inference_time
            )

            plot_data.append(
                {
                    "id": res["response"]["id"],
                    "provider": provider,
                    "time_elapsed": waiting_time,
                    "Breakdown": "Wait",
                }
            )
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
                    "time_elapsed": res["response"]["response"]["measure"][
                        "prepare_time"
                    ],
                    "Breakdown": "Prepare",
                }
            )
            plot_data.append(
                {
                    "id": res["response"]["id"],
                    "provider": provider,
                    "time_elapsed": res["response"]["response"]["measure"][
                        "inference_time"
                    ],
                    "Breakdown": "Inference",
                }
            )

    df = pd.DataFrame(plot_data)
    print(df)

    fig = px.bar(df, x="provider", y="time_elapsed", facet_col="id", color="Breakdown")
    fig.update_layout(
        width=800, height=600, title_x=0.5, title_text="Breakdown of Latency (s)"
    )
    fig.write_image("artifact/results/latency.png", scale=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    args = parser.parse_args()
    plot(args)

    