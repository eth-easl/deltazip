python cli/compress.py --target-model .cache/raw_models/gsd/step_133 --base-model EleutherAI/pythia-2.8b-deduped --dataset .cache/datasets/negotiation_strategy_detection.train.jsonl --bits 4 --sparsity 0.5 --lossless gdeflate --delta subtract --outdir .cache/compressed_models/p2.8b_gsd_133 