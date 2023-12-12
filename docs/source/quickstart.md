
# Quick Start

## Installation

We strongly recommend using Docker to install deltazip, with the following command:

```bash
docker pull xzyaoi/deltazip:0.0.1
```

Deltazip requires NVIDIA GPU to run. Please configure your docker environment according to the [official document](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) and [stackoverflow](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).

The following examples assume that you have installed deltazip in this way.

## Example: Compressing Vicuna-7B

First create a directory to store the cache files:

```bash
mkdir .cache
```

```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e PYTHONPATH=/app \
-e HUGGING_FACE_HUB_TOKEN={PUT_YOUR_TOKEN_HERE} \
-e HF_HOME=/hf_home \
-it -v $PWD/.cache:/cache -v ~/.cache/huggingface:/hf_home \
xzyaoi/deltazip:0.0.1 \
python cli/compress.py --target-model lmsys/vicuna-7b-v1.5 --outdir /cache/compressed_vicuna --dataset /cache/lmsys_512.jsonl --n-samples 256 --bits 2 --sparsity 0.5 --lossless gdeflate --base-model meta-llama/Llama-2-7b-hf --perc-damp 0.01 --block-size 128 --shuffle-dataset --fast-tokenizer --delta subtract
```