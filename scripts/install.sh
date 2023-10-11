conda install -c "nvidia/label/cuda-11.8.0" cuda-runtime
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
pip install -r requirements.txt
pip install transformers sentencepiece accelerate xformers
pip install .