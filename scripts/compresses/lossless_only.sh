python cli/lossless_compress.py --target-model .cache/raw_models/gsd/global_step19 --base-model EleutherAI/pythia-2.8b-deduped --dataset .cache/datasets/negotiation_strategy_detection.train.jsonl --bits 16 --sparsity 0 --lossless gdeflate --delta subtract --outdir .cache/compressed_models/p2.8b_gsd_133 --group-size 128