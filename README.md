# FMZip

FMZip is a library for lossless and lossy compression of foundation models.

It supports (or will support) the following algorithms:

- [GPT-Q](https://arxiv.org/abs/2210.17323)
- [SparseGPT](https://arxiv.org/pdf/2301.00774.pdf)
- [GPT-Zip](https://openreview.net/forum?id=hO0c2tG2xL)
- [GDeflate (by nvcomp)](https://developer.nvidia.com/nvcomp)

## Lossless Comression

## Lossy Compression

## Credits

Heavily inspired by [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)

## Papers

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