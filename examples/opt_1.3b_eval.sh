export PYTHONPATH=.
python cli/ni_main.py \
    --base-model facebook/opt-1.3b \
    --delta-path .cache/compressed_models/answer_verification-2bit-1024g-0.95s-delta \
    --out-dir .cache/results/