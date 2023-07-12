# python cli/benchmark_nvcomp.py --file .cache/model_delta.safetensors --model-type facebook/opt-1.3b --compressed-output .cache/model_delta.fmzip --tensor-shapes .cache/model_delta.shapes

# python cli/benchmark_nvcomp.py --file .cache/7b_model_delta.safetensors --model-type nlpulse/gpt-j-6b-english_quotes --compressed-output .cache/7b_model_delta.fmzip

python cli/e2e_benchmark_nvcomp.py --file .cache/model_delta.safetensors --model-type facebook/opt-1.3b --compressed-output .cache/model_delta.fmzip --tensor-shapes .cache/model_delta.shapes