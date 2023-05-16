from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

ds = Dataset.from_dict({"text":["text me when you", "hello world"]})
tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
valenc = tokenizer(' '.join(ds['text']), return_tensors='pt')
print(valenc.input_ids)
print(valenc.input_ids.shape)