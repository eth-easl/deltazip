ts -S 4
export NCCL_P2P_DISABLE=1

ts -G 1 lm_eval \
    --use_cache .local/cache/vicuna-7b-v1.5 \
    --model_args pretrained=lmsys/vicuna-7b-v1.5,dtype=float16 \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa,gsm8k,truthfulqa \
    --output_path .local/eval_results/vicuna-7b-v1.5 \
    --batch_size 4

ts -G 1 lm_eval \
    --use_cache .local/cache/vicuna-7b-v1.5-2b50s128g \
    --model_args pretrained=.local/merged_models/vicuna-7b-v1.5-2b50s128g,dtype=float16 \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa,gsm8k,truthfulqa \
    --output_path .local/eval_results/vicuna-7b-v1.5-2b50s128g \
    --batch_size 4

ts -G 1 lm_eval \
    --use_cache .local/cache/Llama-2-7b-hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype=float16 \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa,gsm8k,truthfulqa \
    --output_path .local/eval_results/Llama-2-7b-hf-full \
    --batch_size 4

# ts -G 2 lm_eval --model vllm \
#     --model_args pretrained=.local/merged_models/vicuna-13b-v1.5-2b50s128g,dtype=auto,tensor_parallel_size=2 \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa \
#     --output_path .local/eval_results/vicuna-13b-v1.5-2b50s128g \
#     --batch_size 1

# ts -G 2 lm_eval --model vllm \
#     --model_args pretrained=lmsys/vicuna-13b-v1.5,dtype=auto,tensor_parallel_size=2 \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa \
#     --output_path .local/eval_results/vicuna-13b-v1.5-full \
#     --batch_size 1

# ts -G 2 lm_eval --model vllm \
#     --model_args pretrained=.local/merged_models/Llama-2-13b-chat-hf-2b50s128g,dtype=auto,tensor_parallel_size=2 \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa \
#     --output_path .local/eval_results/Llama-2-13b-chat-hf-2b50s128g \
#     --batch_size 1

# ts -G 2 lm_eval --model vllm \
#     --model_args pretrained=meta-llama/Llama-2-13b-hf,dtype=auto,tensor_parallel_size=2 \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa \
#     --output_path .local/eval_results/Llama-2-13b-chat-hf-full \
#     --batch_size 1

# ts -G 2 lm_eval --model vllm \
#     --model_args pretrained=meta-llama/Llama-2-13b-chat-hf,dtype=auto,tensor_parallel_size=2 \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa \
#     --output_path .local/eval_results/Llama-2-13b-chat-hf-full \
#     --batch_size 1

# ts -G 2 lm_eval --model vllm \
#     --model_args pretrained=meta-llama/Llama-2-13b-hf,dtype=auto,tensor_parallel_size=2 \
#     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,hendrycksTest*,sciq,logiqa \
#     --output_path .local/eval_results/Llama-2-13b-hf-full \
#     --batch_size 1