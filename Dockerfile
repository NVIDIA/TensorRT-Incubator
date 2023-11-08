FROM nvidia/cuda:12.0.0-devel-ubuntu22.04
WORKDIR /tripy

LABEL maintainer="NVIDIA CORPORATION"

SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
ENV DEBIAN_FRONTEND=noninteractive 

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
RUN pip install .[docs,dev,test] --extra-index-url https://download.pytorch.org/whl/cu118

RUN mkdir -p /usr/lib/mlir-tensorrt/
COPY mlir-tensorrt/build/lib/Integrations /usr/lib/mlir-tensorrt/
ENV LD_LIBRARY_PATH=/usr/lib/mlir-tensorrt//PJRT/:/usr/local/cuda/lib64/:/usr/local/cuda/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
ENV CUDA_PATH=/usr/local/cuda

########################################
# Build StableHLO python packages
########################################
RUN python3 -m pip install pybind11
RUN apt-get update && \
    apt-get install -y clang ccache git-all lld ninja-build
# Reuse the llvm-project repo from mlir-tensorrt
# Trade-off is llvm-project commit may not be synced with stablehlo/build_tools/llvm_version.txt
RUN mkdir -p /usr/lib/llvm-project/
COPY mlir-tensorrt/third_party/llvm-project /usr/lib/llvm-project/
RUN cd /usr/lib/ && \
    git clone https://github.com/openxla/stablehlo && \
    cd stablehlo && \
    mkdir python-build
RUN cd /usr/lib/stablehlo && build_tools/github_actions/ci_build_cmake_python_api.sh /usr/lib/llvm-project python-build $PWD
ENV PYTHONPATH=/usr/lib/stablehlo/python-build/tools/stablehlo/python_packages/stablehlo/:$PYTHONPATH
