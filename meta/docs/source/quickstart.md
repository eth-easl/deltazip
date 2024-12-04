
# Quick Start

## Installation

We strongly recommend using Docker to install deltazip, with the following command:

```bash
docker pull xzyaoi/deltazip:0.0.1
```

Deltazip requires NVIDIA GPU to run. Please configure your docker environment according to the [official document](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) and [stackoverflow](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).

The following examples assume that you have installed deltazip in this way.

## Example: Compressing Vicuna-7B

We first compress the delta between Vicuna-7B and its pre-trained counterpart (meta-llama/Llama-2-7b-hf) to 2 bits + 50% sparsity:

First create a directory to store the cache files:

```bash
mkdir .cache
```

You will need >= 24GB GPU memory to run the following command:

```sh
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e PYTHONPATH=/app \
-e HUGGING_FACE_HUB_TOKEN={PLACE_YOUR_TOKEN_HERE} \
-e HF_HOME=/hf_home \
-it -v $PWD/.cache:/cache -v ~/.cache/huggingface:/hf_home \
xzyaoi/deltazip:0.0.1 \
python cli/compress.py --target-model lmsys/vicuna-7b-v1.5 --outdir /cache/compressed_vicuna --dataset /cache/lmsys.jsonl --n-samples 256 --bits 2 --sparsity 0.5 --lossless gdeflate --delta subtract --base-model meta-llama/Llama-2-7b-hf --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128
```

Then we can inspect the size post-compression:
```sh
ls -alh .cache/compressed_vicuna/
```

It should be around 1.4GB, so 10x compression compared with fp16, not too bad. Let's see the model quality then. We can evaluate the quality of the compressed model:

```sh
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e PYTHONPATH=/app \
-e HUGGING_FACE_HUB_TOKEN=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 \
-e HF_HOME=/hf_home \
-it -v $PWD/.cache:/cache -v ~/.cache/huggingface:/hf_home \
xzyaoi/deltazip:0.0.1 \
python cli/chat.py --base-model meta-llama/Llama-2-7b-hf --model-path /cache/compressed_vicuna
```

This command starts an interactive chatbot session. You can type in any text and see the response from the model. Example output is shown below:

```
[deltazip] Loading base model...
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.12s/it]
[deltazip] Loading target model...
2023-12-13 02:07:35.541 | INFO     | deltazip.modeling._base:from_compressed:740 - compress config: BaseCompressionConfig(bits=2, sparsity=0.5, prunen=0, prunem=0, group_size=-1, group_rows=-1, block_size=128, damp_percent=0.01, desc_act=True, sym=False, true_sequential=True, lossless='gdeflate', dtype='fp16')
2023-12-13 02:07:35.571 | INFO     | deltazip.modeling._base:from_compressed:786 - lm_head not been quantized, will be ignored when make_quant.
2023-12-13 02:07:38.796 | INFO     | deltazip.modeling._utils:unpack_model:136 - Unpacking model...
2023-12-13 02:07:54.914 | INFO     | deltazip.modeling._utils:unpack_model:140 - Model unpacked.
[deltazip] models loaded
User: What can we do in AI research to address climate change?
Assistant:  There are several ways that AI research can be applied to address climate change. The first is through developing models that can predict the impact of climate change on different ecosystems and species. This can help inform conservation efforts and help protect endangered species.

Another area where AI can be useful is in developing renewable energy sources. For example, AI can be used to optimize the design of solar panels or wind turbines to maximize their efficiency.

AI can also be used to improve the efficiency of transportation and logistics, which can reduce greenhouse gas emissions. For example...
```


