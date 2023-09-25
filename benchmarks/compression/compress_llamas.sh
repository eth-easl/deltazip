# python cli/compress.py --target-model lmsys/vicuna-7b-v1.5 --outdir .cache/compressed_models/vicuna-7b-v1.5 --dataset .cache/datasets/lmsys.jsonl --n-samples 256 --bits 3 --group-size 128 --sparsity 0.5 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-7b-hf

# python cli/compress.py --target-model meta-llama/Llama-2-7b-chat-hf --outdir .cache/compressed_models/llama-2-7b-chat --dataset .cache/datasets/meta.jsonl --n-samples 256 --bits 3 --group-size 128 --sparsity 0.5 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-7b-hf

# python cli/compress.py --target-model Xwin-LM/Xwin-LM-7B-V0.1 --outdir .cache/compressed_models/xwin-lm-7b-v0.1 --dataset .cache/datasets/lmsys.jsonl --n-samples 256 --bits 3 --group-size 128 --sparsity 0.5 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-7b-hf

python cli/compress.py --target-model migtissera/Synthia-7B-v1.2 --outdir .cache/compressed_models/synthia-7b-v1.2 --dataset .cache/datasets/lmsys.jsonl --n-samples 256 --bits 3 --group-size 128 --sparsity 0.5 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-7b-hf
