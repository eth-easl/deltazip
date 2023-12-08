import torch
import transformers
from typing import List, Tuple
from deltazip.pipelines.deltazip_pipeline import DeltaZipPipeline
from deltazip.pipelines.hf_pipeline import HuggingFacePipeline

AVAILABLE_PROVIDERS = ["hf", "deltazip"]


class TextGenerationPipeline:
    def __init__(
        self,
        provider: str,
        provider_args: dict,
    ) -> None:
        self.provider = provider.lower()
        self.deltazip_pipeline = None
        if self.provider.lower() not in AVAILABLE_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}, supported providers are {AVAILABLE_PROVIDERS}"
            )
        if self.provider == "deltazip":
            self.deltazip_pipeline = DeltaZipPipeline(**provider_args)
