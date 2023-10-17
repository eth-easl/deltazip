python cli/lossless_compress.py --target-model FlagAlpha/Llama2-Chinese-13b-Chat --outdir .cache/compressed_models/13b-parameters/lossless/llama-2-chinese-13b-chat-lossless --dataset .cache/datasets/dialogs.jsonl --n-samples 384 --bits 3 --group-size 128 --sparsity 0.3 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-13b-hf

python cli/lossless_compress.py --target-model lmsys/vicuna-13b-v1.5 --outdir .cache/compressed_models/13b-parameters/lossless/lmsys/vicuna-13b-v1.5-v1.5-lossless --dataset .cache/datasets/dialogs.jsonl --n-samples 384 --bits 3 --group-size 128 --sparsity 0.3 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-13b-hf

python cli/lossless_compress.py --target-model meta-llama/Llama-2-13b-chat-hf --outdir .cache/compressed_models/13b-parameters/lossless/llama-2-13b-chat-lossless --dataset .cache/datasets/dialogs.jsonl --n-samples 384 --bits 3 --group-size 128 --sparsity 0.3 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-13b-hf

python cli/lossless_compress.py --target-model Xwin-LM/Xwin-LM-13B-V0.1 --outdir .cache/compressed_models/13b-parameters/lossless/xwin-lm-13b-v0.1-lossless --dataset .cache/datasets/dialogs.jsonl --n-samples 384 --bits 3 --group-size 128 --sparsity 0.3 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-13b-hf

python cli/lossless_compress.py --target-model migtissera/Synthia-13B-v1.2 --outdir .cache/compressed_models/13b-parameters/lossless/synthia-13b-v1.2-lossless --dataset .cache/datasets/dialogs.jsonl --n-samples 384 --bits 3 --group-size 128 --sparsity 0.3 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-13b-hf

