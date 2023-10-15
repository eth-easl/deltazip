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
        backend = item['backend']
        backend_name = backend['name']
        if backend_name == "hf":
            backend_name = f"{backend_name}, bsz={backend['args']['batch_size']}, max_models={backend['args']['max_num_models']}"
        elif backend_name == "fmzip":
            backend_name = f"{backend_name}, bsz={backend['args']['batch_size']}, max_deltas={backend['args']['max_num_deltas']}, strategy={backend['args']['placement_strategy']}, lossless_only={backend['args']['lossless_only']}"
        
        min_length = item['gen_args']['min_length']
        for idx, query in enumerate(item['results']):
            print(query)
            #total_elapsed = query['total_elapsed']
            tokenize_time = query['measure']['tokenize_time']
            loading_time = query['measure']['loading_time']
            prepare_time = query['measure']['prepare_time']
            inference_time = query['measure']['inference_time']
            # waiting_time = total_elapsed - (tokenize_time + loading_time + prepare_time + inference_time)
            # plot_data.append({
            #     "id": idx,
            #     "backend": backend_name,
            #     "tokens": min_length,
            #     "time_elapsed": waiting_time,
            #     "Breakdown": "Others",
            # })
            plot_data.append({
                "id": idx,
                "backend": backend_name,
                "tokens": min_length,
                "time_elapsed": tokenize_time,
                "Breakdown": "Tokenize",
            })
            plot_data.append({
                "id": idx,
                "backend": backend_name,
                "tokens": min_length,
                "time_elapsed": loading_time,
                "Breakdown": "Loading",
            })
            plot_data.append({
                "id": idx,
                "backend": backend_name,
                "tokens": min_length,
                "time_elapsed": prepare_time,
                "Breakdown": "Prepare",
            })
            plot_data.append({
                "id": idx,
                "backend": backend_name,
                "tokens": min_length,
                "time_elapsed": inference_time,
                "Breakdown": "Generation",
            })
    df = pd.DataFrame(plot_data)
    print(df)
    fig = px.bar(df, x="backend", y="time_elapsed", facet_col="id", color="Breakdown")
    fig.update_layout(
        width=800, height=600, title_x=0.5, title_text="Breakdown of Latency (s)"
    )
    fig.write_image(args.output, scale=2)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="artifact/results/latency.json")
    parser.add_argument("--output", type=str, default="artifact/plots/latency.png")
    args = parser.parse_args()
    plot(args)