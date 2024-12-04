---
hide-toc: true
---

# DeltaZip

[![Paper](https://img.shields.io/badge/arxiv-2312.05215-blue)]([https://](https://arxiv.org/abs/2312.05215))  [![Documents](https://img.shields.io/badge/docs-in_progress-gren)](https://deltazip.readthedocs.io/en/latest/) [![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/eth-easl/deltazip/blob/init/LICENSE)

DeltaZip is a system for compressing and serving **full-parameter** fine-tuned LLMs.

## Abstract

Fine-tuning large language models (LLMs) for downstream tasks can greatly improve model quality, however serving many different fine-tuned LLMs concurrently for users in multi-tenant environments is challenging. Dedicating GPU memory for each model is prohibitively expensive and naively swapping large model weights in and out of GPU memory is slow. Our key insight is that fine-tuned models can be quickly swapped in and out of GPU memory by extracting and compressing the delta between each model and its pre-trained base model. 

We propose DeltaZip, an LLM serving system that efficiently serves multiple full-parameter fine-tuned models concurrently by aggressively compressing model deltas by a factor of 6x to 8x while maintaining high model quality. DeltaZip increases serving throughput by 1.5x to 3x and improves SLO attainment compared to a vanilla HuggingFace serving system.

## Quick Start

[Quick Start Guide](/quickstart)


## Acknowledgements

Heavily inspired by

* [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
* [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt)
* [PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
* [qwopqwop200/GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)

## Citation

If you found this code useful, please cite our paper:

```
@article{yao2023deltazip,
  title={DeltaZip: Multi-Tenant Language Model Serving via Delta Compression},
  author={Yao, Xiaozhe and Klimovic, Ana},
  journal={arXiv preprint arXiv:2312.05215},
  year={2023}
}
```
```
@inproceedings{
    isik2023gptzip,
    title={{GPT}-Zip: Deep Compression of Finetuned Large Language Models},
    author={Berivan Isik and Hermann Kumbong and Wanyi Ning and Xiaozhe Yao and Sanmi Koyejo and Ce Zhang},
    booktitle={Workshop on Efficient Systems for Foundation Models @ ICML2023},
    year={2023},
    url={https://openreview.net/forum?id=hO0c2tG2xL}
}
```

## Related Projects

- [FMEngine](https://fmengine.readthedocs.io/en/latest/): Utilities for training large language models.

```{toctree}
:caption: Getting Started
:hidden:

quickstart
```