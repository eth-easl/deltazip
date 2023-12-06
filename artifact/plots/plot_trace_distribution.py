import pandas as pd
import plotly.express as px
import datasets
from datetime import datetime

ds = datasets.load_dataset("lmsys/chatbot_arena_conversations", "train")

reqs = []
timestamps = []
models = []
timewindow = 5
time_grid = []
for h in range(0, 24):
    for m in range(0, 60, timewindow):
        time_grid.append(f"{h:02d}:{m:02d}")
for d in ds["train"]:
    time = datetime.fromtimestamp(d["tstamp"])
    # only keep the minute
    date = time.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")
    time = time.replace(year=1970, month=1, day=1).strftime("%H:%M")
    time = min(
        time_grid,
        key=lambda x: abs(
            datetime.strptime(x, "%H:%M") - datetime.strptime(time, "%H:%M")
        ),
    )
    time = date + " " + time
    model_a = d["model_a"]
    model_b = d["model_b"]
    models.append(model_a)
    models.append(model_b)
    reqs.append({"date": date, "time": time, "model": model_a})
    reqs.append({"date": date, "time": time, "model": model_b})
# aggregate reqs manually
reformatted_reqs = []
for req in reqs:
    date = req["date"]
    time = req["time"]
    model = req["model"]
    # check if we have a req with the same date, time, and model
    # if so, increment the count
    found = False
    for r in reformatted_reqs:
        if r["date"] == date and r["time"] == time and r["model"] == model:
            r["count"] += 1
            found = True
    if not found:
        reformatted_reqs.append(
            {"date": date, "time": time, "model": model, "count": 1}
        )
df = pd.DataFrame(reformatted_reqs)
print(df.head())
# select only time in May 7 to May 14
sub_df = df[(df["date"] >= "2023-06-11") & (df["date"] <= "2023-06-18")]
new_df = sub_df.pivot(index="model", columns="time")["count"].fillna(0)
fig = px.imshow(
    new_df,
    x=new_df.columns,
    y=new_df.index,
    color_continuous_scale="inferno_r",
    aspect="auto",
)
fig.update_layout(
    title={
        "xanchor": "center",
        "x": 0.5,
        "font_size": 28,
    }
)
fig.update_xaxes(
    title_text="",
    tickangle=30,
    title_standoff=25,
    title_font=dict(size=28),
    tickfont=dict(size=26),
)
fig.update_yaxes(
    title_text="",
    title_standoff=25,
    title_font=dict(size=28),
    tickfont=dict(size=24),
)
fig.update_layout(
    font_family="Arial",
    font_color="black",
    title_font_family="Arial",
    title_font_color="black",
    legend_title_font_color="black",
)
fig.update_layout(
    font=dict(size=24),
    title=dict(font=dict(size=28)),
    legend=dict(font=dict(size=28)),
    legend_title=dict(font=dict(size=28)),
)
fig.update_layout(
    width=1200, height=1000, title_x=0.5, margin=dict(l=40, r=40, t=80, b=80)
)
fig.write_image("artifact/results/images/trace.png", scale=4)
