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
    apt-get install -y software-properties-common sudo fakeroot python3-pip && \
    apt-get clean && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install poetry


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
RUN poetry config virtualenvs.create false && poetry install --with docs --with dev --with test && poetry cache clear --all . -n && rm -rf /root/.cache/pypoetry/artifacts

