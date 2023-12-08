import json
import torch
import random
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def get_jsonl(
    train_path,
    val_path,
    n_samples,
    seed,
    seq_len,
    model_name,
    val_size=None,
    val_seq_len=256,
    padding=False,
):
    """
    train_path: path to train jsonl file
    test_path: path to test jsonl file
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    with open(train_path, "r") as f:
        traindata = [json.loads(line) for line in f.readlines()]
    with open(val_path, "r") as f:
        valdata = [json.loads(line) for line in f.readlines()]
    traindata = {"text": [d["text"] for d in traindata]}
    valdata = {"text": [d["text"] for d in valdata]}
    traindata = Dataset.from_dict(traindata)
    valdata = Dataset.from_dict(valdata)
    set_seed(seed)

    trainloader = []
    for _ in range(n_samples):
        # for all datasets, we take the samples that are longer than seq_len
        while True:
            i = random.randint(0, len(traindata) - 1)
            if padding:
                trainenc = tokenizer(
                    traindata[i]["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=seq_len,
                    return_tensors="pt",
                )
            else:
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seq_len:
                break
        if not padding:
            # then clip the samples to seq_len
            i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
            j = i + seq_len
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        else:
            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    if val_size is not None:
        valenc = tokenizer(" ".join(valdata[:val_size]["text"]), return_tensors="pt")
    else:
        valenc = tokenizer(" ".join(valdata["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (val_seq_len * seq_len)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=""):
    if name.startswith("ni_"):
        taskname = name.replace("ni_", "")
    return get_jsonl(
        f".cache/ni_calib/train/{taskname}.jsonl",
        f".cache/ni_calib/test/{taskname}.jsonl",
        n_samples=nsamples,
        seed=seed,
        seq_len=seqlen,
        model_name=model,
        val_size=1000,
        padding=True,
    )
