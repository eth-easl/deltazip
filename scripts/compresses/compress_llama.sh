# python cli/compress.py --base-model meta-llama/Llama-2-7b-hf --target-model meta-llama/Llama-2-7b-chat-hf --dataset /home/xzyao/Documents/datasets/ni_calib/train/answer_verification.jsonl --bits 4 --sparsity 0.8 --outdir .cache/compressed_models/llama-2-7b --delta subtract --lossless gdeflate

python cli/compress.py --base-model openlm-research/open_llama_3b_v2 --target-model xzyao/openllama-3b-chat --dataset .cache/datasets/dialogs.jsonl --bits 2 --sparsity 0 --outdir .cache/compressed_models/3b-parameters/2bits-openllama --delta subtract --lossless gdeflate --n-samples 512

python cli/compress.py --base-model openlm-research/open_llama_3b_v2 --target-model /mnt/scratch/xiayao/cache/experiments/fmzip/finetuned_raw/llama-3b/task372_synthetic_palindrome_numbers/global_step105/ --dataset /mnt/scratch/xiayao/cache/datasets/qi/ar/task372_synthetic_palindrome_numbers.train.jsonl --bits 16 --sparsity 0 --outdir .cache/16bits-delta-debug-1024s/ --delta subtract --lossless gdeflate --n-samples 1024 --perc-damp 0.0001

python cli/lossless_compress.py --base-model openlm-research/open_llama_3b_v2 --target-model /mnt/scratch/xiayao/cache/experiments/fmzip/finetuned_raw/llama-3b/task372_synthetic_palindrome_numbers/global_step105/ --dataset /mnt/scratch/xiayao/cache/datasets/qi/ar/task372_synthetic_palindrome_numbers.train.jsonl --bits 16 --sparsity 0 --outdir .cache/lossless-delta-debug-1024s/ --delta subtract --lossless gdeflate --n-samples 1024

python cli/ni_evaluate_debug.py --base-model openlm-research/open_llama_3b_v2 --target-model .cache/lossless-delta-debug-1024s --delta subtract --input-file /mnt/scratch/xiayao/cache/datasets/qi/test/task372_synthetic_palindrome_numbers.test.jsonl --input-field input --max-length 32 --output-file /mnt/scratch/xiayao/cache/experiments/fmzip/generation_llama/new_generation_results_4b0s/task372_synthetic_palindrome_numbers/global_step105.jsonl