# Mixed Precision Generation
from fmzip.pipelines import TextGenerationPipeline

pipeline = TextGenerationPipeline("EleutherAI/pythia-2.8b-deduped")
pipeline.load_delta(".cache/compressed_models/p2.8b_gsd_133")
