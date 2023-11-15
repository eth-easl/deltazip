# python cli/compress.py --base-model meta-llama/Llama-2-7b-hf --target-model meta-llama/Llama-2-7b-chat-hf --dataset /home/xzyao/Documents/datasets/ni_calib/train/answer_verification.jsonl --bits 4 --sparsity 0.8 --outdir .cache/compressed_models/llama-2-7b --delta subtract --lossless gdeflate

python cli/compress.py --base-model openlm-research/open_llama_3b_v2 --target-model xzyao/openllama-3b-chat --dataset .cache/datasets/dialogs.jsonl --bits 2 --sparsity 0 --outdir .cache/compressed_models/3b-parameters/2bits-openllama --delta subtract --lossless gdeflate --n-samples 512

python cli/lossless_compress.py --base-model openlm-research/open_llama_3b_v2 --target-model xzyao/openllama-3b-chat --dataset .cache/datasets/dialogs.jsonl --bits 16 --sparsity 0 --outdir .cache/compressed_models/3b-parameters/openllama-3b-chat-lossless --delta subtract --lossless gdeflate --n-samples 128

