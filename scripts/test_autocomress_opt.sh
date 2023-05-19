export PYTHONPATH=.
python src/main.py \
    --model-type opt \
    --base-model 'facebook/opt-1.3b' \
    --target-model '.cache/models/answer_verification' \
    --dataset ni_answer_verification \
    --wbits 2 3 4 \
    --sparsities 0.1 0.33 0.5 0.67 0.9 0.95 0.99 \
    --tol 0.1 \
    --n-samples 128 \
    --seed 42