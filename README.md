# FMZip

## Compressing

Pre-set wbit and sparsity.

```bash
export PYTHONPATH=.
python cli/delta_preset.py \
    --base-model facebook/opt-1.3b \
    --target-model .cache/models/answer_verification \
    --dataset .cache/ni_calib/train/answer_verification.jsonl \
    --out-dir .cache/compressed_models/ \
    --wbit 2 \
    --sparsity 0.95 \
    --n-samples 128
```
Note: Sparsity is not in yet.


Auto-tuned.

TBA

## Evaluation

### On Natural Instructions

```bash
export PYTHONPATH=.
python cli/ni_main.py \
    --base-model facebook/opt-1.3b \
    --delta-path .cache/compressed_models/answer_verification-2bit-1024g-0.95s-delta \
    --out-dir .cache/results/
```

This will evaluate the base model + delta on all natural instruction selections.

(we probably need a conditional perplexity as metric - for now we leveraged the eval code from natural instructions repo)

### On LM-Eval-Harness

TBA

### On Perplexity

TBA