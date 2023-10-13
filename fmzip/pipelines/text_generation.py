import torch
import transformers
from typing import List, Tuple
from fmzip.pipelines.pipeline import FMZipPipeline

AVAILABLE_PROVIDERS = ["hf", "fmzip"]


class TextGenerationPipeline:
    def __init__(
        self,
        provider: str,
        provider_args: dict,
    ) -> None:
        self.provider = provider.lower()
        self.fmzip_pipeline = None
        if self.provider.lower() not in AVAILABLE_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}, supported providers are {AVAILABLE_PROVIDERS}"
            )
        if self.provider == "fmzip":
            self.fmzip_pipeline = FMZipPipeline(**provider_args)
