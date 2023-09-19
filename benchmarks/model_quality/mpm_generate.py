import os

job = f"python cli/mpm_generate.py --base-model EleutherAI/pythia-2.8b-deduped --target-model .cache/compressed_models/p2.8b_gsd_133 --input-file .cache/datasets/negotiation_strategy_detection.train.jsonl --input-field text --max-length 32 --output-file .cache/output.jsonl --batch-size 4"

os.system(job)