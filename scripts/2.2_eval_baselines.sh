echo "Evaluating SparseGPT"

docker run \
--rm \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 /deltazip/cli/restore_sparsegpt.py --target-model /local/sparsegpt_models/lmsys.vicuna-7b-v1.5.4b_2n4m_128bs --outdir /local/merged_models/sparsegpt.lmsys.vicuna-7b-v1.5.4b_2n4m_128bs

docker run \
--rm \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 -m lm_eval --model hf --model_args pretrained=/local/merged_models/sparsegpt.lmsys.vicuna-7b-v1.5.4b_2n4m_128bs/restored_.local.sparsegpt_models.lmsys.vicuna-7b-v1.5.4b_2n4m_128bs --tasks boolq,truthfulqa,logiqa --output_path /local/eval_results/sparsegpt.lmsys.vicuna-7b-v1.5.4b128g --batch_size auto --trust_remote_code

echo "Evaluating AWQ"

docker run \
--rm \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 -m lm_eval --model hf --model_args pretrained=/local/awq_models/awq.lmsys.vicuna-7b-v1.5.4b128g,dtype=auto --tasks boolq,truthfulqa,logiqa --output_path /local/eval_results/awq.lmsys.vicuna-7b-v1.5.4b128g --batch_size auto --trust_remote_code