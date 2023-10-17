conda install -c "nvidia/label/cuda-11.8.0" cuda-runtime
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install -c anaconda cmake
pip install -r requirements.txt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip install transformers sentencepiece accelerate xformers
pip install --no-deps .