import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('.cache/results/cr.csv')
print(df)
sns.set(font_scale=2)

sns.set_theme(style="whitegrid")
ax = sns.scatterplot(data=df, x="model_size", y="compression_rate", hue="comp_type", style="target_model", s=200)
ax.set(xlabel="Model size (GB)", ylabel="Compression rate")
# make legend outside of the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# make plot larger
plt.gcf().set_size_inches(8, 4)
plt.tight_layout()
plt.savefig(".cache/plots/compression_rate.pdf")
