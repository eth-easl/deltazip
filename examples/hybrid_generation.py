# Mixed Precision Generation
from fmzip.pipelines import MixedPrecisionModel

mpm = MixedPrecisionModel("EleutherAI/pythia-2.8b-deduped")
mpm.load_delta(".cache/compressed_models/p2.8b_gsd_133")

results = mpm.generate([
    ("Computer Science is about ", ".cache/compressed_models/p125m_gsd_133"), 
    ("What is the weather today?", ".cache/compressed_models/p125m_gsd_133"),
    ("Alan Turing is ", ".cache/compressed_models/p125m_gsd_133"), 
    ("QED is ", ".cache/compressed_models/p125m_gsd_133")
])

print(results)