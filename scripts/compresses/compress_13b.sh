python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model meta-llama/Llama-2-13b-chat-hf --outdir .local/compressed_models/meta-llama.Llama-2-13b-chat-hf.2b50s128g --dataset .local/datasets/meta.jsonl --n-samples 256 --bits 2 --sparsity 0.5 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model meta-llama/Llama-2-13b-chat-hf --outdir .local/compressed_models/meta-llama.Llama-2-13b-chat-hf.2b75s128g --dataset .local/datasets/meta.jsonl --n-samples 256 --bits 2 --sparsity 0.75 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model meta-llama/Llama-2-13b-chat-hf --outdir .local/compressed_models/meta-llama.Llama-2-13b-chat-hf.2b90s128g --dataset .local/datasets/meta.jsonl --n-samples 256 --bits 2 --sparsity 0.9 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model meta-llama/Llama-2-13b-chat-hf --outdir .local/compressed_models/meta-llama.Llama-2-13b-chat-hf.4b50s128g --dataset .local/datasets/meta.jsonl --n-samples 256 --bits 4 --sparsity 0.5 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model meta-llama/Llama-2-13b-chat-hf --outdir .local/compressed_models/meta-llama.Llama-2-13b-chat-hf.4b75s128g --dataset .local/datasets/meta.jsonl --n-samples 256 --bits 4 --sparsity 0.75 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model meta-llama/Llama-2-13b-chat-hf --outdir .local/compressed_models/meta-llama.Llama-2-13b-chat-hf.4b90s128g --dataset .local/datasets/meta.jsonl --n-samples 256 --bits 4 --sparsity 0.9 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model lmsys/vicuna-13b-v1.5 --outdir .local/compressed_models/lmsys.vicuna-13b-v1.5.4b50s128g --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 4 --sparsity 0.5 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model lmsys/vicuna-13b-v1.5 --outdir .local/compressed_models/lmsys.vicuna-13b-v1.5.4b75s128g --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 4 --sparsity 0.75 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model lmsys/vicuna-13b-v1.5 --outdir .local/compressed_models/lmsys.vicuna-13b-v1.5.4b90s128g --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 4 --sparsity 0.9 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model lmsys/vicuna-13b-v1.5 --outdir .local/compressed_models/lmsys.vicuna-13b-v1.5.2b75s128g --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 2 --sparsity 0.75 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model lmsys/vicuna-13b-v1.5 --outdir .local/compressed_models/lmsys.vicuna-13b-v1.5.2b90s128g --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 2 --sparsity 0.9 --lossless gdeflate --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128