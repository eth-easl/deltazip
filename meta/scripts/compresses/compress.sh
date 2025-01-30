python cli/compress.py --base-model meta-llama/Llama-3.2-1B --target-model meta-llama/Llama-3.2-1B-Instruct --outdir .local/compressed_models/ --n-samples 256 --bits 4 --prunen 2 --prunem 4 --lossless none --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model meta-llama/Llama-3.2-3B --target-model meta-llama/Llama-3.2-3B-Instruct --outdir .local/compressed_models/ --n-samples 256 --bits 4 --prunen 2 --prunem 4 --lossless none --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model meta-llama/Llama-3.1-8B --target-model meta-llama/Llama-3.1-8B-Instruct --outdir .local/compressed_models/ --n-samples 256 --bits 4 --prunen 2 --prunem 4 --lossless none --delta subtract --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model meta-llama/Meta-Llama-3-8B --target-model meta-llama/Meta-Llama-3-8B-Instruct --outdir .local/compressed_models/ --n-samples 256 --bits 4 --prunen 2 --prunem 4 --lossless none --delta subtract --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model google/gemma-2-2b --target-model google/gemma-2-2b-it --outdir .local/compressed_models/ --n-samples 256 --bits 4 --prunen 2 --prunem 4 --lossless none --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model google/gemma-2-9b --target-model google/gemma-2-9b-it --outdir .local/compressed_models/ --n-samples 256 --bits 4 --prunen 2 --prunem 4 --lossless none --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model meta-llama/Llama-3.2-1B --target-model meta-llama/Llama-3.2-1B-Instruct --outdir .local/compressed_models/ --n-samples 256 --bits 2 --prunen 2 --prunem 4 --lossless none --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model meta-llama/Llama-3.2-3B --target-model meta-llama/Llama-3.2-3B-Instruct --outdir .local/compressed_models/ --n-samples 256 --bits 2 --prunen 2 --prunem 4 --lossless none --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model meta-llama/Llama-3.1-8B --target-model meta-llama/Llama-3.1-8B-Instruct --outdir .local/compressed_models/ --n-samples 256 --bits 2 --prunen 2 --prunem 4 --lossless none --delta subtract --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model meta-llama/Meta-Llama-3-8B --target-model meta-llama/Meta-Llama-3-8B-Instruct --outdir .local/compressed_models/ --n-samples 256 --bits 2 --prunen 2 --prunem 4 --lossless none --delta subtract --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model google/gemma-2-2b --target-model google/gemma-2-2b-it --outdir .local/compressed_models/ --n-samples 256 --bits 2 --prunen 2 --prunem 4 --lossless none --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

python cli/compress.py --base-model google/gemma-2-9b --target-model google/gemma-2-9b-it --outdir .local/compressed_models/ --n-samples 256 --bits 2 --prunen 2 --prunem 4 --lossless none --delta subtract  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym