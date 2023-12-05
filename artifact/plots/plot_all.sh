python artifact/plots/plot_ablation_sequence.py
python artifact/plots/plot_trace_distribution.py
python artifact/plots/plot_combined_ni_model_quality.py --input artifact/results/quality/ni/results.csv --output artifact/results/images/ni_quality.png
python artifact/plots/plot_combined_latency_breakdown.py --input artifact/results/poisson/3b --output artifact/results/images/3b_latency.png
python artifact/plots/plot_combined_throughput.py --input artifact/results/poisson/3b --output artifact/results/images/3b_throughput.png
python artifact/plots/plot_combined_slo.py --input artifact/results/poisson/3b --output artifact/results/images/3b_slo.png