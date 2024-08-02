ARG BASE_IMAGE
FROM ${BASE_IMAGE}

#############################
# Basic Dependencies
#############################

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
        wget gnupg apt-utils curl \
        default-jdk-headless default-jre-headless \
        git python3.10 python3.10-distutils python3.10-dev python3.10-venv \
        libprotobuf-dev protobuf-compiler && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

########################################
# Install OpenMP & other dependencies
########################################
RUN apt-get update && apt-get install -y \
    cuda-nsight-systems-${CUDA_VERSION%.*} \
    openmpi-bin openmpi-common libopenmpi-dev && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# The LLVM version we should download for the compilation toolchain
ARG LLVM_VERSION=17
ENV LLVM_VERSION=$LLVM_VERSION
COPY build_tools/scripts/install_recommended_build_tools.sh /tmp/install_tools.sh
RUN chmod +x /tmp/install_tools.sh && \
    /tmp/install_tools.sh && \
    rm /tmp/install_tools.sh


# Install CUDNN for building with older TRT versions.
RUN apt-get update && apt-get install -y \
     libsndfile1 libcudnn8 && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install extra python development requirements.
COPY python/requirements-dev.txt /tmp/requirements-dev.txt
COPY python/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements-dev.txt && \
    rm /tmp/requirements-dev.txt /tmp/requirements.txt
