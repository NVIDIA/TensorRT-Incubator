FROM nvidia/cuda:12.2.0-base-ubuntu22.04
WORKDIR /tripy

LABEL maintainer="NVIDIA CORPORATION"

SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000

RUN groupadd -r -f -g ${gid} trtuser && \
    useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser && \
    usermod -aG sudo trtuser && \
    echo 'trtuser:nvidia' | chpasswd && \
    mkdir -p /workspace && chown trtuser /workspace && \
    apt-get update && \
    apt-get install -y software-properties-common sudo fakeroot python3-pip gdb && \
    apt-get clean && \
    python3 -m pip install --upgrade pip


# Install the recommended version of TensorRT for development.
ARG CUDNN_VERSION=8.9.2.26-1+cuda12.1
ARG TRT_VERSION=8.6.1.6-1+cuda12.0
RUN apt-get update && \
    apt-get install -y \
    tensorrt-libs=${TRT_VERSION} \
    tensorrt-dev=${TRT_VERSION} && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


COPY pyproject.toml /tripy/pyproject.toml
RUN pip install .[docs,dev,test] --extra-index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN mkdir -p /usr/lib/mlir-tensorrt/
COPY mlir-tensorrt/build/lib/Integrations /usr/lib/mlir-tensorrt/
ENV LD_LIBRARY_PATH=/usr/lib/mlir-tensorrt//PJRT/:/usr/local/cuda/lib64/:/usr/local/cuda-12.2/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
