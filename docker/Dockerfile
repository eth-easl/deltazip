FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV MAX_JOBS=16
# https://github.com/oobabooga/text-generation-webui/issues/2128
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX;8.9;9.0"
COPY . /app
WORKDIR /app
RUN pip install -r /app/requirements.txt
RUN pip install --no-deps .
RUN pip install triton
RUN mv /usr/local/cuda/include/cufile.h /tmp/cufile.h && git clone --branch branch-23.10 https://github.com/rapidsai/kvikio.git && cd kvikio && ./build.sh kvikio && cd .. && rm -rf kvikio && mv /tmp/cufile.h /usr/local/cuda/include/cufile.h
RUN git clone https://github.com/xzyaoi/triton.git -b feat/singularity && cd triton/python && pip install cmake && pip install -e .