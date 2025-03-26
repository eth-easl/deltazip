docker run \
--rm \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 /deltazip/cli/restore.py --target-model /local/compressed_models/lmsys.vicuna-7b-v1.5.2b_2n4m_128bs --outdir /local/merged_models

docker run \
--rm \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 -m lm_eval --model hf --model_args pretrained=/local/merged_models/merged_.local.compressed_models.lmsys.vicuna-7b-v1.5.2b_2n4m_128bs,dtype=auto --tasks boolq,truthfulqa,logiqa --output_path /local/eval_results/lmsys.vicuna-7b-v1.5.2b_2n4m_128bs --batch_size auto --trust_remote_code