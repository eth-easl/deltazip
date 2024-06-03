export TS_VISIBLE_DEVICES=2,3
ts -S 2
ts -G 1 python cli/compress.py --base-model meta-llama/Llama-2-7b-hf --target-model lmsys/vicuna-7b-v1.5 --outdir .local/compressed_models/sparsegpt.lmsys.vicuna-7b-v1.5.4b75s128g --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 4 --sparsity 0.75 --lossless gdeflate  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model lmsys/vicuna-13b-v1.5 --outdir .local/compressed_models/sparsegpt.lmsys.vicuna-13b-v1.5.4b75s128g --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 4 --sparsity 0.75 --lossless gdeflate  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128

python cli/compress.py --base-model meta-llama/Llama-2-7b-hf --target-model lmsys/vicuna-7b-v1.5 --outdir .local/compressed_models/lmsys.vicuna-7b-v1.5.4b75s128g.sym --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 2 --sparsity 0.75 --lossless gdeflate  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --delta subtract --sym

python cli/compress.py --base-model meta-llama/Llama-2-7b-hf --target-model lmsys/vicuna-7b-v1.5 --outdir .local/compressed_models/lmsys.vicuna-7b-v1.5.4b75s128gn2m4.sym. --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 4 --sparsity 0.75 --lossless gdeflate  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --delta subtract --sym --prunen 2 --prunem 4

python cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model lmsys/vicuna-13b-v1.5 --outdir .local/compressed_models/lmsys.vicuna-13b-v1.5.4bn2m4.sym. --dataset .local/datasets/lmsys.jsonl --n-samples 256 --bits 4 --sparsity 0.75 --lossless gdeflate  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --delta subtract --sym --prunen 2 --prunem 4