import torch
import transformers
from typing import List, Tuple
from fmzip.pipelines.hf_pipeline import HuggingFacePipeline
from fmzip.pipelines.fmzip_pipeline import FMZipPipeline

class InferenceService:
    def __init__(self, base_model: str, backend, backend_args) -> None:
        self.backend = backend
        self.base_model = base_model
        self.backend_args = backend_args
        if backend == "hf":
            self.pipeline = HuggingFacePipeline(
                base_model
                **backend_args
            )
        elif backend == 'fmzip':
            self.pipeline = FMZipPipeline(
                base_model=base_model,
                **backend_args
            )
    
    def generate(self, queries: List[Tuple]):
        if self.backend == 'hf':
            results = self.pipeline.generate()