strategy_mapping = {"none": "None", "addback": "AS", "colocate": "MMA"}
project_name = "FiniCompress"


def get_provider_name(provider):
    if provider["name"] == "hf":
        return "HuggingFace"
    elif provider["name"] == "fmzip":
        name = f"{project_name}<br>bsz={provider['args'].get('batch_size', 1)}<br>{strategy_mapping[provider['args'].get('placement_strategy','none')]}, {'Lossless' if provider['args'].get('lossless_only', False) else 'Lossy'}"
        if provider["args"].get("kernel", "") == "triton":
            name += ", Triton"
        return name


def get_provider_order(provider):
    if provider["name"] == "hf":
        return str(999)
    elif provider["name"] == "fmzip":
        if (
            provider["args"].get("placement_strategy", "none") == "colocate"
            and provider["args"].get("batch_size", 4) == 4
        ):
            return str(0)
        if (
            provider["args"].get("placement_strategy", "none") == "colocate"
            and provider["args"].get("batch_size", 1) == 1
        ):
            return str(1)
        if (
            provider["args"].get("placement_strategy", "none") == "addback"
            and provider["args"].get("lossless_only", False) == False
        ):
            return str(2)
        if (
            provider["args"].get("placement_strategy", "none") == "addback"
            and provider["args"].get("lossless_only", True) == True
        ):
            return str(3)


def set_plotly_theme(fig):
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    return fig


def set_font(fig):
    fig.update_layout(
        font_family="Arial",
        font_color="black",
        title_font_family="Arial",
        title_font_color="black",
        legend_title_font_color="black",
        title=dict(font=dict(size=36)),
        legend=dict(font=dict(size=32)),
        legend_title=dict(font=dict(size=28)),
    )
    fig.update_xaxes(title=dict(font=dict(size=28)), tickfont_size=30)
    fig.update_yaxes(title=dict(font=dict(size=28)), tickfont_size=30)
    return fig
