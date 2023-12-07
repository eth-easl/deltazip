import os
import json
import pandas as pd
import plotly.express as px

from artifact.plots.utils import get_provider_name, get_provider_order, set_plotly_theme

bsz = 2
tokens = [64, 128, 256, 512, 1024, 2048, 3072]
plot_data = []
provider_mapping = {
    "FiniCompress<br>bsz=1<br>AS, Lossy": "AS, Lossy, bsz=1",
    "FiniCompress<br>bsz=2<br>MMA, Lossy": "MMA, Lossy, bsz=2",
    "FiniCompress<br>bsz=2<br>MMA, Lossy, Triton": "MMA, Lossy, bsz=2, Triton",
}
for token in tokens:
    with open(
        f"artifact/results/ablation/sequence/bsz=2/{token}tokens.json", "r"
    ) as fp:
        results = json.load(fp)
    # calculate
    for item in results:
        provider = get_provider_name(item["system"])
        provider = provider_mapping[provider]
        total_jobs = len(item["results"])
        last_job = max(
            item["results"], key=lambda x: x["time_elapsed"] + x["relative_start_at"]
        )
        throughput = total_jobs / (
            last_job["time_elapsed"] + last_job["relative_start_at"]
        )
        plot_data.append(
            {"tokens": token, "provider": provider, "throughput": throughput}
        )

symbols = ['square', 'circle', 'diamond', 'cross']
df = pd.DataFrame(plot_data)
fig = px.line(df, x="tokens", y="throughput", color="provider", markers=True, symbol="provider", symbol_sequence=symbols)

fig.update_traces(line=dict(width=5))
fig.update_traces(marker={"size": 20})
fig.update_xaxes(
    title_text="Output Sequence Length (tokens)",
    title_font=dict(size=36),
    tickfont_size=24,
)
fig.update_yaxes(
    title_text="Throughput (requests/s)", title_font=dict(size=36), tickfont_size=24
)

fig.update_layout(
    width=1200,
    height=600,
    title_x=0.5,
    title_text=f"Throughput of Different Backends",
)
fig.update_layout(
    title=dict(font=dict(size=36)),
    legend=dict(font=dict(size=24)),
    legend_title=dict(font=dict(size=28), text="Backend"),
)
fig.update_layout(
    font_family="Arial",
    font_color="black",
    title_font_family="Arial",
    title_font_color="black",
    legend_title_font_color="black",
)
fig.update_layout(legend=dict(
    font=dict(size=28),
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99,
    bgcolor="white",
    
))
fig.update_xaxes(type="log")
fig = set_plotly_theme(fig)
fig.write_image("artifact/results/images/sequence.png", scale=2)
