# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

ARG BASE_IMG=ubuntu:24.04
FROM ${BASE_IMG} AS dev-base

# Specify user IDs
ARG GROUP
ARG GID
ARG USER
ARG UID

# Run below commands as root
USER root

# Install packages
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -P /tmp/ && \
    dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && \
    rm /tmp/cuda-keyring_1.1-1_all.deb

RUN apt-get update && \
    apt-get install -y \
    lld \
    clang \
    clang-format \
    gdb \
    black \
    python3-dev \
    cmake \
    sudo \
    vim \
    ninja-build \
    patch \
    pybind11-dev \
    git \
    libnccl-dev \
    libopenmpi-dev \
    openmpi-bin \
    python3-pip \
    python3-pynvml \
    python3-numpy \
    python3-psutil \
    cuda-toolkit-12 && \
    pip install --break-system-packages nanobind cupy-cuda12x


# Set workdir before launching container
WORKDIR /opt/src/mlir-tensorrt

# Add user permissions
RUN groupadd -o -g ${GID} ${GROUP} && \
    useradd -u ${UID} -g ${GROUP} -ms /bin/bash ${USER} && \
    usermod -aG sudo ${USER} && \
    chown -R ${USER}:${GROUP} /opt/src/mlir-tensorrt && \
    echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to user
USER ${USER}
