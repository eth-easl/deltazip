python cli/ni_evaluate_lessmem.py --base-model meta-llama/Llama-2-7b-hf --target-model .cache/compressed_models/7b-parameters/bits-2/vicuna-7b-v1.5 --delta subtract --input-file artifact/data/lmsys.jsonl --input-field text --output-file artifact/results/lmsys_output.jsonl --temperature 0.6 --top-k 50 --do-sample --max-length 512

python cli/ni_evaluate_lessmem.py --base-model meta-llama/Llama-2-7b-hf --target-model .cache/compressed_models/7b-parameters/bits-2/xwin-lm-7b-v0.1 --delta subtract --input-file artifact/data/lmsys.jsonl --input-field text --output-file artifact/results/lmsys_output_xwin_2bit.jsonl --temperature 0.6 --top-k 50 --do-sample --max-length 512

python cli/ni_evaluate_nocompress.py --target-model Xwin-LM/Xwin-LM-7B-V0.1 --input-file artifact/data/lmsys.jsonl --input-field text --output-file artifact/results/lmsys_output_xwin_16bit.jsonl --temperature 0.6 --top-k 50 --do-sample --max-length 512

python cli/ni_evaluate_nocompress.py --target-model lmsys/vicuna-7b-v1.5 --input-file artifact/data/lmsys.jsonl --input-field text --output-file artifact/results/lmsys_output_vicuna_16bit.jsonl --temperature 0.6 --top-k 50 --do-sample --max-length 512

python cli/ni_evaluate.py --base-model openlm-research/open_llama_3b_v2 --target-model .cache/test --delta subtract --input-file /mnt/scratch/xiayao/cache/datasets/qi/test/task065_timetravel_consistent_sentence_classification.test.jsonl --input-field input --max-length 32 --output-file .cache/task065.jsonl