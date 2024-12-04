import os

template = """---
datasets:
- {dataset_id}
base_model:
- {model_id}
library_name: transformers, deltazip
---

## {model_id} - {scheme} Compression

This is a compressed model using [deltazip](https://github.com/eth-easl/deltazip).

[Paper](https://arxiv.org/abs/2312.05215), [Compression Tool](https://github.com/eth-easl/deltazip), [Inference Engine (Soon)](https://github.com/eth-easl/deltazip).

## Compression Configuration

- Base Model: {model_id}
- Compression Scheme: {scheme}
- Dataset: {dataset_id}
- Dataset Split: {ds_split}
- Max Sequence Length: {seq_len}
- Number of Samples: {n_samples}

## Sample Output

#### Prompt: 

```
{prompt}
```

#### Output: 

```
{output}
```

## Evaluation

<TODO>

"""

def generate_readme(config) -> str:
    readme = template.format(
        model_id=config['model_id'],
        scheme=config['scheme'],
        dataset_id=config['dataset_id'],
        ds_split=config['ds_split'],
        seq_len=config['seq_len'],
        n_samples=config['n_samples'],
        prompt=config['prompt'],
        output=config['output']
    )
    return readme

def upload_and_delete(org_id, model_id, local_path):
    cmd = f"huggingface-cli upload {org_id}/{model_id} {local_path} --repo-type model"
    os.system(cmd)
    os.system(f"rm -rf {local_path}")
