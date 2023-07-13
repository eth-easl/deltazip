# python cli/benchmark_nvcomp.py --file .cache/model_delta.safetensors --model-type facebook/opt-1.3b --compressed-output .cache/model_delta.fmzip --tensor-shapes .cache/model_delta.shapes

# python cli/benchmark_nvcomp.py --file .cache/7b_model_delta.safetensors --model-type nlpulse/gpt-j-6b-english_quotes --compressed-output .cache/7b_model_delta.fmzip

# python cli/e2e_benchmark_nvcomp.py --file .cache/model_delta.safetensors --base-model facebook/opt-1.3b --compressed-output .cache/model_delta.fmzip --tensor-shapes .cache/model_delta.shapes --target-model facebook/opt-iml-max-1.3b

python cli/e2e_benchmark_nvcomp.py --base-model EleutherAI/gpt-j-6b --compressed-output .cache/compressed_model/nlpulse-gpt-j-6b-english_quotes/compressed_model.safetensors --tensor-shapes .cache/compressed_model/nlpulse-gpt-j-6b-english_quotes/tensor_shapes.json --target-model nlpulse/gpt-j-6b-english_quotes