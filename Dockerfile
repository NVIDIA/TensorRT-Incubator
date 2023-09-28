FROM nvcr.io/nvidia/pytorch:23.06-py3
# FROM nvcr.io/nvidia/jax:23.08-py3
WORKDIR /tripy


LABEL maintainer="NVIDIA CORPORATION"
 
SHELL ["/bin/bash", "-c"]
 
# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace
 
RUN pip install poetry

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    fakeroot
 
# Install PyPI packages
RUN pip3 install --upgrade pip
