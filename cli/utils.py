template = """---
datasets:
- {dataset_id}
base_model:
- {model_id}
library_name: transformers
---

## {model_id} - {scheme} Compression

This is a compressed model using [deltazip](https://github.com/eth-easl/deltazip).

## Compression Configuration

- Base Model: {model_id}
- Compression Scheme: {scheme}
- Dataset: {dataset_id}
- Dataset Split: {dataset_split}
- Number of Samples: {n_samples}
- Preprocessor: {preprocessor}
- Maximum Sequence Length: {max_seq_length}

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
        dataset_split=config['dataset_split'],
        n_samples=config['n_samples'],
        preprocessor=config['preprocessor'],
        max_seq_length=config['max_seq_length'],
        prompt=config['prompt'],
        output=config['output']
    )
    return readme