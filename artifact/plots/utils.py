

strategy_mapping = {"none": "None", "addback": "AS", "colocate": "MMA"}

def get_provider_name(provider):
    if provider["name"] == "hf":
        return "HuggingFace"
    elif provider["name"] == "fmzip":
        return f"DeltaZip, bsz={provider['args'].get('batch_size', 1)}<br>{strategy_mapping[provider['args'].get('placement_strategy','none')]} {'Lossless' if provider['args'].get('lossless_only', False) else 'Lossy'}"

def get_provider_order(provider):
    if provider["name"] == "hf":
        return str(999)
    elif provider["name"] == "fmzip":
        if provider['args'].get('placement_strategy','none') == 'colocate' and provider['args'].get('batch_size', 4)==4:
            return str(0)
        if provider['args'].get('placement_strategy','none') == 'colocate' and provider['args'].get('batch_size', 1)==1:
            return str(1)
        if provider['args'].get('placement_strategy','none') == 'addback' and provider['args'].get('lossless_only', False)==False:
            return str(2)
        if provider['args'].get('placement_strategy','none') == 'addback' and provider['args'].get('lossless_only', True)==True:
            return str(3)