mkdir -p $WORKDIR/results/

docker run \
--rm \
--runtime=nvidia \
-p 8000:8000 \
-e PYTHONPATH=/deltazip:/workspace \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
--shm-size=10.24gb \
-e HOST_IP=127.0.0.1 \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 /workspace/vllm/entrypoints/openai/api_server.py --model meta-llama/Llama-2-13b-hf --tensor-parallel-size 4 --swap-space 1 --enforce-eager --disable-custom-all-reduce --gpu-memory-utilization 0.85 --max-deltas 6 --max-cpu-deltas 16 --enable-delta --delta-modules delta-1=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.0 delta-2=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.1 delta-3=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.2 delta-4=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.3 delta-5=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.4 delta-6=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.5 delta-7=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.6 delta-8=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.7 delta-9=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.8 delta-10=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.9 delta-11=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.10 delta-12=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.11 delta-13=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.12 delta-14=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.13 delta-15=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.14 delta-16=/local/models/deltas/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs.15