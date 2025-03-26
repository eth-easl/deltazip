mkdir -p $WORKDIR/converted_weights/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs

docker run \
--runtime=nvidia \
-e PYTHONPATH=/deltazip \
-e HF_TOKEN=$HF_TOKEN \
-e HF_HOME=/HF_HOME \
-v $(realpath $WORKDIR):/local \
-v $HF_HOME/:/HF_HOME \
ghcr.io/xiaozheyao/deltazip:0.0.1 \
python3 /workspace/triteia/triteia/tools/converters/convert_deltazip.py --ckpt /local/compressed_models/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs/deltazip-compressed.safetensors --tp-size 4 --save-path /local/converted_weights/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs/deltazip-compressed.safetensors --lossless

cp $WORKDIR/compressed_models/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs/*.json $WORKDIR/converted_weights/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs

cp $WORKDIR/compressed_models/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs/*.model $WORKDIR/converted_weights/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs

python3 scripts/helpers/copy_weights.py --source $WORKDIR/converted_weights/lmsys.vicuna-13b-v1.5.4b_2n4m_128bs --target $WORKDIR/models/deltas --num-copies 24