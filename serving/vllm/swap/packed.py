import torch
from typing import Optional, List


class ModelLayerWeights:
    def __init__(
        self,
        module_name: str,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        self.module_name = module_name
        self.weight = weight
        self.bias = bias


class PackedModelLayerWeights(ModelLayerWeights):
    def __init__(
        self,
        module_name: str,
        weights: List[torch.Tensor],
        biases: List[torch.Tensor],
    ) -> None:
        super().__init__(module_name=module_name, weight=weights, bias=biases)

    @classmethod
    def pack(cls, models: List["ModelLayerWeights"]) -> "PackedModelLayerWeights":
        first_model = next(model for model in models if model is not None)
        module_name = first_model.module_name
        return cls(
            module_name=module_name,
            weights=[model.weight for model in models],
            biases=[model.bias for model in models],
        )

    @property
    def input_dim(self) -> int:
        raise NotImplementedError()

    @property
    def output_dim(self) -> int:
        raise NotImplementedError()

    @property
    def is_packed(self) -> bool:
        return True
