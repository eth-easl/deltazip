import torch
import transformers
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import count_nonzero
from deltazip.utils.plot_utils import set_matplotlib_style
cmp = sns.color_palette("tab10")
def calculate_sparsity(tensor):
    return 1.0 - (count_nonzero(tensor) / float(tensor.size))

base_model = transformers.AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16
)
finetuned_model = transformers.AutoModel.from_pretrained(
    "lmsys/vicuna-7b-v1.5", torch_dtype=torch.float16
)
keyword = f'layers.10.self_attn.q_proj.weight'
plot_finetuned_weight = (
    finetuned_model.state_dict()[keyword].flatten().cpu().numpy()
)
plot_pretrained_weight = base_model.state_dict()[keyword].flatten().cpu().numpy()
sample_size = 5000
sampled_indices = np.random.choice(
    len(plot_finetuned_weight), size=sample_size, replace=False
)
plot_finetuned_weight = plot_finetuned_weight[sampled_indices]
plot_pretrained_weight = plot_pretrained_weight[sampled_indices]
plot_delta_weight = plot_finetuned_weight - plot_pretrained_weight

set_matplotlib_style()

fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, constrained_layout=True, figsize=(9, 3.75), sharex=True)
ax1.plot(plot_pretrained_weight, color=cmp[0], alpha=0.8)
ax2.plot(plot_finetuned_weight,  color=cmp[1], alpha=0.8)
ax3.plot(plot_delta_weight,      color=cmp[2], alpha=0.8)
ax1.set_ylabel(f"Base")
ax2.set_ylabel(f"FMT")
ax3.set_ylabel(f"Delta")

ax1.set_ylim(-0.1, 0.1)
ax1.grid(axis="x", linestyle=":")
ax2.set_ylim(-0.1, 0.1)
ax2.grid(axis="x", linestyle=":")
ax3.set_ylim(-0.1, 0.1)
ax3.grid(axis="x", linestyle=":")

fig.suptitle(f"{keyword}")

sns.despine()
fig.savefig("plot_dist.pdf", bbox_inches="tight")
