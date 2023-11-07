FROM xzyaoi/fmsys:0.0.9

ENV CUDA_HOME=/usr/local/cuda
ENV CPATH=$CUDA_HOME/include:$CPATH
ENV MAX_JOBS=16
# https://github.com/oobabooga/text-generation-webui/issues/2128
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX;8.9;9.0"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install --no-deps .
# update pytorch to nightly version
# for load_state_dict(assign=True)