FROM ubuntu:22.04

LABEL org.opencontainers.image.description="Tripy development container"

WORKDIR /tripy

SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd -r -f -g ${gid} trtuser && \
    useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser && \
    usermod -aG sudo trtuser && \
    echo 'trtuser:nvidia' | chpasswd && \
    mkdir -p /workspace && chown trtuser /workspace

RUN apt-get update && \
    apt-get install -y sudo python3 python3-pip gdb git wget curl && \
    apt-get clean && \
    python3 -m pip install --upgrade pip

COPY .lldbinit /root/
COPY pyproject.toml /tripy/pyproject.toml

RUN pip install build .[docs,dev,test] \
    -f https://nvidia.github.io/TensorRT-Incubator/packages.html \
    --extra-index-url https://download.pytorch.org/whl \
    --extra-index-url https://pypi.nvidia.com

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

# Export tripy into the PYTHONPATH so it doesn't need to be installed after making changes
ENV PYTHONPATH=/tripy
