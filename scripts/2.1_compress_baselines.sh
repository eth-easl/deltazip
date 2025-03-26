echo "Compressing with SparseGPT"

docker run \
--rm \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN= \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 /deltazip/cli/compress.py --target-model lmsys/vicuna-7b-v1.5 --outdir /local/sparsegpt_models --n-samples 256 --bits 4 --prunen 2 --prunem 4 --lossless gdeflate  --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym

echo "Compressing with AWQ"

docker run \
--rm \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 /deltazip/cli/compress_awq.py --model lmsys/vicuna-7b-v1.5 --outdir /local/awq_models --disable-safetensors

echo "Compressing baselines done!"