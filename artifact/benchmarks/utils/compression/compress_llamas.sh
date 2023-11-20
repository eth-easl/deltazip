python cli/compress.py --target-model FlagAlpha/Llama2-Chinese-7b-Chat --outdir .cache/compressed_models/7b-parameters/2bits/llama2-chinese-7b-chat --dataset .cache/datasets/meta.jsonl --n-samples 256 --bits 2 --sparsity 0 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-7b-hf --shuffle-dataset

python cli/chat.py --target-model .cache/compressed_models/7b-parameters/2bits/llama2-chinese-7b-chat --base-model meta-llama/Llama-2-7b-hf --delta subtract --fast-tokenizer

python cli/compress.py --target-model xzyao/openllama-3b-chat --outdir .cache/compressed_models/3b-parameters/2bits/openllama-chat --dataset .cache/datasets/dialogs.jsonl --n-samples 256 --bits 4 --sparsity 0 --lossless gdeflate --delta subtract --base-model openlm-research/open_llama_3b_v2 --perc-damp 0.01 --block-size 128 --shuffle-dataset --fast-tokenizer

python cli/chat.py --target-model .cache/compressed_models/3b-parameters/2bits/openllama-chat-sparsegpt --base-model openlm-research/open_llama_3b_v2 --delta subtract --fast-tokenizer

python cli/compress.py --target-model xzyao/openllama-3b-chat --outdir .cache/compressed_models/3b-parameters/2bits/openllama-chat-sparsegpt --dataset .cache/datasets/dialogs.jsonl --n-samples 256 --bits 4 --sparsity 0 --lossless gdeflate --base-model openlm-research/open_llama_3b_v2 --perc-damp 0.01 --block-size 128 --shuffle-dataset --fast-tokenizer

# python cli/compress.py --target-model FlagAlpha/Llama2-Chinese-7b-Chat --outdir .cache/compressed_models/nodelta/llama2-chinese-7b-chat --dataset .cache/datasets/meta.jsonl --n-samples 256 --bits 2 --group-size 128 --sparsity 0.3 --lossless gdeflate --base-model meta-llama/Llama-2-7b-hf

# python cli/compress.py --target-model lmsys/vicuna-7b-v1.5 --outdir .cache/compressed_models/nodelta/vicuna-7b-v1.5 --dataset .cache/datasets/lmsys.jsonl --n-samples 256 --bits 2 --group-size 128 --sparsity 0.3 --lossless gdeflate --base-model meta-llama/Llama-2-7b-hf

# python cli/compress.py --target-model meta-llama/Llama-2-7b-chat-hf --outdir .cache/compressed_models/nodelta/llama-2-7b-chat --dataset .cache/datasets/meta.jsonl --n-samples 256 --bits 2 --group-size 128 --sparsity 0.3 --lossless gdeflate --base-model meta-llama/Llama-2-7b-hf

# python cli/compress.py --target-model Xwin-LM/Xwin-LM-7B-V0.1 --outdir .cache/compressed_models/nodelta/xwin-lm-7b-v0.1 --dataset .cache/datasets/lmsys.jsonl --n-samples 256 --bits 2 --group-size 128 --sparsity 0.3 --lossless gdeflate --base-model meta-llama/Llama-2-7b-hf

# python cli/compress.py --target-model migtissera/Synthia-7B-v1.2 --outdir .cache/compressed_models/nodelta/synthia-7b-v1.2 --dataset .cache/datasets/lmsys.jsonl --n-samples 256 --bits 2 --group-size 128 --sparsity 0.3 --lossless gdeflate --base-model meta-llama/Llama-2-7b-hf

# python cli/compress.py --target-model xzyao/openllama-3b-chat --outdir .cache/compressed_models/3b-parameters/openllama-chat-3bit --dataset .cache/datasets/lmsys.jsonl --n-samples 256 --bits 3 --group-size 128 --sparsity 0.3 --lossless gdeflate --delta subtract --base-model openlm-research/open_llama_3b_v2

python cli/ni_eval_debug.py --base-model openlm-research/open_llama_3b_v2 --target-model .cache/compressed_models/3b-parameters/2bits/openllama-chat --delta subtract --input-file /nfs/cache/datasets/qi/test/task151_tomqa_find_location_easy_clean.test.jsonl --input-field input --max-length 32 --fast-tokenizer --temperature 0.1