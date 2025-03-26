docker run \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 /deltazip/cli/compress.py --base-model meta-llama/Llama-2-13b-hf --target-model lmsys/vicuna-13b-v1.5 --outdir /local/compressed_models --n-samples 256 --bits 4 --prunen 2 --prunem 4 --lossless gdeflate --delta subtract --shuffle-dataset --fast-tokenizer --perc-damp 0.01 --block-size 128 --sym