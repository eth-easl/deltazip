docker run --gpus all -it \
--volume .:/app \
--ulimit memlock=-1 --ulimit stack=67108864 \
-p 8000:8000 \
--volume /home/ubuntu/.cache/huggingface/:/hf_cache \
--env HF_HOME=/hf_cache \
--env HF_TOKEN=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 \
xzyaoi/fmzip:0.0.1 \
uvicorn fmzip.rest.server:app --host 0.0.0.0