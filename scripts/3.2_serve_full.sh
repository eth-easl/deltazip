mkdir -p $WORKDIR/results/

docker run \
--rm \
--runtime=nvidia \
--shm-size=10.24gb \
-p 8000:8000 \
-e PYTHONPATH=/deltazip:/workspace \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-e RAY_memory_monitor_refresh_ms=0 \
-e HOST_IP=127.0.0.1 \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 /workspace/vllm/entrypoints/openai/api_server.py --model meta-llama/Llama-2-13b-hf --tensor-parallel-size 4 --swap-space 1 --enforce-eager --disable-custom-all-reduce --gpu-memory-utilization 0.85 --max-deltas 0 --max-cpu-deltas 0 --max-swap-slots 1 --max-cpu-models 4 --enable-swap --swap-modules delta-1=/local/models/full/vicuna-13b-v1.5.0 delta-2=/local/models/full/vicuna-13b-v1.5.1 delta-3=/local/models/full/vicuna-13b-v1.5.2 delta-4=/local/models/full/vicuna-13b-v1.5.3 delta-5=/local/models/full/vicuna-13b-v1.5.4 delta-6=/local/models/full/vicuna-13b-v1.5.5 delta-7=/local/models/full/vicuna-13b-v1.5.6 delta-8=/local/models/full/vicuna-13b-v1.5.7 delta-9=/local/models/full/vicuna-13b-v1.5.8 delta-10=/local/models/full/vicuna-13b-v1.5.9 delta-11=/local/models/full/vicuna-13b-v1.5.10 delta-12=/local/models/full/vicuna-13b-v1.5.11 delta-13=/local/models/full/vicuna-13b-v1.5.12 delta-14=/local/models/full/vicuna-13b-v1.5.13 delta-15=/local/models/full/vicuna-13b-v1.5.14 delta-16=/local/models/full/vicuna-13b-v1.5.15