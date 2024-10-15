ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Specify user IDs
ARG GROUP
ARG GID
ARG USER
ARG UID

# Run below commands as root
USER root

#############################
# Basic Dependencies
#############################

ARG DEBIAN_FRONTEND=noninteractive

RUN <<EOF
set -e
apt-get update
apt-get install -y \
    wget gnupg apt-utils curl \
    default-jdk-headless default-jre-headless \
    git python3.10 python3.10-distutils python3.10-dev python3.10-venv \
    python3-pip libprotobuf-dev protobuf-compiler libsndfile1 libcudnn8 \
    cuda-nsight-systems-${CUDA_VERSION%.*} \
    openmpi-bin openmpi-common libopenmpi-dev
apt-get clean -y
rm -rf /var/lib/apt/lists/*
EOF

#------------------------------------------------------------------------------
# Python setup
# We install pyenv only for rockylinux, but we don't preload dependencies.
# For ubuntu, preload the required python dependencies in the default python
# environment.
#------------------------------------------------------------------------------
ARG PYTHON_VERSION=3.10
ENV PYENV_ROOT="/pyenv"
ENV PATH="/pyenv/bin:/pyenv/shims:$PATH"
COPY python/requirements-dev.txt /tmp/requirements-dev.txt
COPY python/requirements.txt /tmp/requirements.txt
RUN <<EOF
set -e
python3 -m pip install -r /tmp/requirements-dev.txt
# Cleanup dependencies file.
rm /tmp/requirements-dev.txt /tmp/requirements.txt
EOF

#------------------------------------------------------------------------------
# LLVM toolchain installation
# We install LLVM toolchain (clang, lld, etc) for ubuntu. This is a no-op for
# rockylinux, where we just use GNU toolchain already installed above.
#------------------------------------------------------------------------------
ARG LLVM_VERSION=17
ENV LLVM_VERSION=$LLVM_VERSION
COPY build_tools/scripts/install_recommended_build_tools.sh /tmp/install_tools.sh

RUN <<EOF
set -e
chmod +x /tmp/install_tools.sh
/tmp/install_tools.sh
rm /tmp/install_tools.sh
EOF

# Install bazel
ARG ARCH="x86_64"
ARG BAZEL_VERSION=6.4.0
RUN wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${ARCH} -O /usr/bin/bazel \
    && chmod a+x /usr/bin/bazel

# Set workdir before launching container
WORKDIR /opt/src/mlir-tensorrt

# Add user permissions
RUN groupadd -o -g ${GID} ${GROUP} && \
    useradd -u ${UID} -g ${GROUP} -ms /bin/bash ${USER} && \
    usermod -aG sudo ${USER} && \
    chown -R ${USER}:${GROUP} /opt/src/mlir-tensorrt && \
    echo "%sudo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to user
USER ${USER}