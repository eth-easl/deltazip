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
    
def get_wikitext2(nsamples, seed, seqlen, model):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_jsonl(train_path, val_path, n_samples, seed, seq_len, model_name, val_size=None, val_seq_len=256, padding=False):
    """
    train_path: path to train jsonl file
    test_path: path to test jsonl file
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    with open(train_path, 'r') as f:
        traindata = [json.loads(line) for line in f.readlines()]
    with open(val_path, 'r') as f:
        valdata = [json.loads(line) for line in f.readlines()]
    traindata = {"text": [d['text'] for d in traindata]}
    valdata = {"text": [d['text'] for d in valdata]}
    traindata = Dataset.from_dict(traindata)
    valdata = Dataset.from_dict(valdata)
    set_seed(seed)

    trainloader = []
    for _ in range(n_samples):
        # for all datasets, we take the samples that are longer than seq_len
        while True:
            i = random.randint(0, len(traindata) - 1)
            if padding:
                trainenc = tokenizer(traindata[i]['text'], padding='max_length', truncation=True, max_length=seq_len, return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
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
        valenc = tokenizer(' '.join(valdata[:val_size]['text']), return_tensors='pt')
    else:
        valenc = tokenizer(' '.join(valdata['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(val_seq_len * seq_len)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    if name == "answer_verification":
        return get_jsonl(
            ".cache/ni_calib/train/answer_verification.jsonl", 
            ".cache/ni_calib/test/answer_verification.jsonl", 
            nsamples, 
            seed, 
            seqlen, 
            model, 
            val_size=1000, 
            padding=True
        )
    if name == "coherence_classification":
        return get_jsonl(".cache/ni_calib/test/coherence_classification.jsonl", ".cache/ni_calib/test/coherence_classification.jsonl", nsamples, seed, seqlen, model, val_size=1000, padding=True)
    if name == "commonsense_classification":
        return get_jsonl(".cache/ni_calib/train/commonsense_classification.jsonl", ".cache/ni_calib/test/commonsense_classification.jsonl", nsamples, seed, seqlen, model, val_size=1000, padding=True)
    if name == "dialogue_state_tracking":
        return get_jsonl(".cache/ni_calib/train/dialogue_state_tracking.jsonl", ".cache/ni_calib/test/dialogue_state_tracking.jsonl", nsamples, seed, seqlen, model, val_size=1000, padding=True)
    if name == "fact_verification":
        return get_jsonl(".cache/ni_calib/train/fact_verification.jsonl", ".cache/ni_calib/test/fact_verification.jsonl", nsamples, seed, seqlen, model, val_size=1000, padding=True)
    if name == "gender_classification":
        return get_jsonl(".cache/ni_calib/train/gender_classification.jsonl", ".cache/ni_calib/test/gender_classification.jsonl", nsamples, seed, seqlen, model, val_size=1000, padding=True)
    if name == "irony_detection":
        return get_jsonl(".cache/ni_calib/train/irony_detection.jsonl", ".cache/ni_calib/test/irony_detection.jsonl", nsamples, seed, seqlen, model, val_size=1000, padding=True)
    if name == "stance_detection":
        return get_jsonl(".cache/ni_calib/train/stance_detection.jsonl", ".cache/ni_calib/test/stance_detection.jsonl", nsamples, seed, seqlen, model, val_size=1000, padding=True)
    if name == "toxic_language_detection":
        return get_jsonl(".cache/ni_calib/train/toxic_language_detection.jsonl", ".cache/ni_calib/test/toxic_language_detection.jsonl", nsamples, seed, seqlen, model, val_size=1000, padding=True)
    if name == "word_semantics":
        return get_jsonl(
            ".cache/ni_calib/train/word_semantics.jsonl", 
            ".cache/ni_calib/test/word_semantics.jsonl", 
            nsamples, 
            seed, 
            seqlen, 
            model, 
            val_size=1000, 
            padding=True
        )