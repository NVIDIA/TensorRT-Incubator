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
