FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04
WORKDIR /tripy

LABEL maintainer="NVIDIA CORPORATION"

SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
ENV DEBIAN_FRONTEND=noninteractive

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

RUN groupadd -r -f -g ${gid} trtuser && \
    useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser && \
    usermod -aG sudo trtuser && \
    echo 'trtuser:nvidia' | chpasswd && \
    mkdir -p /workspace && chown trtuser /workspace && \
    apt-get update && \
    apt-get install -y software-properties-common sudo fakeroot python3-pip gdb git wget libcudnn8 lldb-15 && \
    apt-get clean && \
    python3 -m pip install --upgrade pip

# Create symbolic link for LLDB Python packages (adjust Python version if necessary)
RUN ln -s /usr/bin/lldb-15 /usr/bin/lldb
RUN ln -s /usr/lib/llvm-15/lib/python3.10/dist-packages/lldb/* /usr/lib/python3/dist-packages/lldb/

# Copy your .lldbinit file into the home directory of the root user
COPY .lldbinit /root/

# Install the recommended version of TensorRT for development.
RUN cd /usr/lib/ && \
    wget http://cuda-repo/release-candidates/Libraries/TensorRT/v10.0/10.0.0.1-a3728acd/12.2-r535/Linux-x64-agnostic/tar/TensorRT-10.0.0.1.Linux.x86_64-gnu.cuda-12.2.tar.gz && \
    tar -xvzf TensorRT-10.0.0.1.Linux.x86_64-gnu.cuda-12.2.tar.gz && \
    rm TensorRT-10.0.0.1.Linux.x86_64-gnu.cuda-12.2.tar.gz
ENV LD_LIBRARY_PATH=/usr/lib/TensorRT-10.0.0.1/lib/:$LD_LIBRARY_PATH

ARG gitlab_user
ARG gitlab_token
COPY pyproject.toml /tripy/pyproject.toml
RUN pip install .[docs,dev,test] \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://${gitlab_user}:${gitlab_token}@gitlab-master.nvidia.com/api/v4/projects/73221/packages/pypi/simple --trusted-host gitlab-master.nvidia.com

########################################
# Configure mlir-tensorrt packages
########################################
# WAR's for small bugs in the MLIR-TRT wheels
# Protobuf isn't actually used for how Tripy uses MLIR-TRT, so we just install any version to make the loader happy.

RUN apt-get install -y libopenmpi3 libopenmpi-dev libprotobuf-dev && \
    ln -snf /usr/lib/x86_64-linux-gnu/libprotobuf.so /usr/lib/x86_64-linux-gnu/libprotobuf.so.29
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/mlir_tensorrt/compiler/_mlir_libs/:$LD_LIBRARY_PATH

# Export tripy into the PYTHONPATH so it doesn't need to be installed after making changes
ENV PYTHONPATH=/tripy:$PYTHONPATH
