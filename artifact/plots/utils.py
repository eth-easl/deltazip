

strategy_mapping = {"none": "None", "addback": "Add-Back", "colocate": "Mixed-Prec"}

def get_provider_name(provider):
    if provider["name"] == "hf":
        return "HuggingFace"
    elif provider["name"] == "fmzip":
        return f"DeltaZip, bsz={provider['args'].get('batch_size', 1)}<br>{strategy_mapping[provider['args'].get('placement_strategy','none')]}<br>{'Lossless' if provider['args'].get('lossless_only', False) else 'Lossy'}"