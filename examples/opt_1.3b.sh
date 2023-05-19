export PYTHONPATH=.
python cli/delta_preset.py \
    --base-model facebook/opt-1.3b \
    --target-model .cache/models/answer_verification \
    --dataset .cache/ni_calib/train/answer_verification.jsonl \
    --out-dir .cache/compressed_models/ \
    --wbit 2 \
    --sparsity 0.95 \
    --n-samples 128