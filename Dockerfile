FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04
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
    apt-get install -y software-properties-common sudo fakeroot python3-pip gdb git wget libcudnn8 && \
    apt-get clean && \
    python3 -m pip install --upgrade pip


# Install the recommended version of TensorRT for development.
RUN cd /usr/lib/ && \
    wget http://cuda-repo/release-candidates/Libraries/TensorRT/v10.0/10.0.0.1-a3728acd/12.2-r535/Linux-x64-agnostic/tar/TensorRT-10.0.0.1.Linux.x86_64-gnu.cuda-12.2.tar.gz && \
    tar -xvzf TensorRT-10.0.0.1.Linux.x86_64-gnu.cuda-12.2.tar.gz && \
    rm TensorRT-10.0.0.1.Linux.x86_64-gnu.cuda-12.2.tar.gz
ENV LD_LIBRARY_PATH=/usr/lib/TensorRT-10.0.0.1/lib/:$LD_LIBRARY_PATH


COPY pyproject.toml /tripy/pyproject.toml
RUN pip install .[docs,dev,test] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --extra-index-url https://download.pytorch.org/whl/cu118

########################################
# Configure StableHLO python packages
########################################
RUN mkdir -p /usr/lib/stablehlo
COPY stablehlo /usr/lib/stablehlo
ENV PYTHONPATH=/usr/lib/stablehlo/python-build/tools/stablehlo/python_packages/stablehlo:$PYTHONPATH

########################################
# Configure mlir-tensorrt packages
########################################
RUN mkdir -p /usr/lib/mlir-tensorrt/
COPY mlir-tensorrt/build/lib/Integrations /usr/lib/mlir-tensorrt/
ENV LD_LIBRARY_PATH=/usr/lib/mlir-tensorrt/PJRT/:/usr/local/cuda/lib64/:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
