FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04
WORKDIR /tripy

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
    apt-get install -y software-properties-common sudo fakeroot python3-pip gdb git wget libcudnn8 curl jq && \
    apt-get clean && \
    python3 -m pip install --upgrade pip

# Copy your .lldbinit file into the home directory of the root user
COPY .lldbinit /root/

# Install the recommended version of TensorRT for development.
RUN cd /usr/lib/ && \
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/tars/TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-12.4.tar.gz && \
    tar -xvzf TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-12.4.tar.gz && \
    rm TensorRT-10.1.0.27.Linux.x86_64-gnu.cuda-12.4.tar.gz
ENV LD_LIBRARY_PATH=/usr/lib/TensorRT-10.1.0.27/lib/:$LD_LIBRARY_PATH

COPY pyproject.toml /tripy/pyproject.toml
RUN pip install .[docs,dev,test] \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    -f https://github.com/NVIDIA/TensorRT-Incubator/releases/ \
    --extra-index-url https://download.pytorch.org/whl


########################################
# Configure mlir-tensorrt packages
########################################
# WAR's for small bugs in the MLIR-TRT wheels
# Protobuf isn't actually used for how TriPy uses MLIR-TRT, so we just install any version to make the loader happy.

RUN apt-get install -y libopenmpi3 libopenmpi-dev libprotobuf-dev && \
    ln -snf /usr/lib/x86_64-linux-gnu/libprotobuf.so /usr/lib/x86_64-linux-gnu/libprotobuf.so.29

# Installl lldb for debugging purposes in Tripy container.
# The LLVM version should correspond on LLVM_VERSION specified in https://github.com/NVIDIA/TensorRT-Incubator/blob/main/mlir-tensorrt/build_tools/docker/Dockerfile#L30.
ARG LLVM_VERSION=17
ENV LLVM_VERSION=$LLVM_VERSION
ENV LLVM_PACKAGES="lldb-${LLVM_VERSION}"
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-$LLVM_VERSION main" > /etc/apt/sources.list.d/llvm.list && \
    echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-$LLVM_VERSION main" >> /etc/apt/sources.list.d/llvm.list && \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key 2>/dev/null > /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y ${LLVM_PACKAGES} && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/lldb-17 /usr/bin/lldb

RUN pip3 install --upgrade build

# Export tripy into the PYTHONPATH so it doesn't need to be installed after making changes
ENV PYTHONPATH=/tripy:$PYTHONPATH
