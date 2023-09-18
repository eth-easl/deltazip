# Mixed Precision Generation
from fmzip.pipelines import MixedPrecisionModel

mpm = MixedPrecisionModel("EleutherAI/pythia-2.8b-deduped")
mpm.load_delta(".cache/compressed_models/p2.8b_gsd_133")

mpm.generate([{"text": "Hello world", "model": ".cache/compressed_models/p2.8b_gsd_133"}])