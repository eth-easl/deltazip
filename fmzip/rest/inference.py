from typing import List
from fmzip.pipelines import MixedPrecisionModel

class InferenceService():
    def __init__(self, provider: str, **kwargs) -> None:
        self.provider = provider
        if provider=='fmzip':
            self.mpm = MixedPrecisionModel(**kwargs)
        else:
            raise NotImplementedError
    
    def generate(self, queries: List):
        queries = [(query.prompt, query.model) for query in queries]
        return self.mpm.generate(queries)