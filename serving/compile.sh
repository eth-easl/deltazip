export TORCH_CUDA_ARCH_LIST='8.0 8.6'
pip install flash_attn==2.5.8
pip install git+https://github.com/xiaozheyao/triteia.git
pip install -r requirements.txt
pip install --no-build-isolation --no-cache-dir .