lm_eval --model vllm \
    --model_args pretrained=.local/merged_models/vicuna-13b-v1.5-2b50s128g,dtype=auto,tensor_parallel_size=2 \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --batch_size 1

# accelerate launch -m lm_eval \
#     --model_args pretrained=lmsys/vicuna-7b-v1.5,dtype=float16 \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
#     --batch_size 4