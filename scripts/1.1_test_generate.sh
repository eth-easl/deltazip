docker run \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 /deltazip/cli/generate.py --target-model /local/compressed_models/lmsys.vicuna-7b-v1.5.2b_2n4m_128bs --prompt "Who is Alan Turing?"