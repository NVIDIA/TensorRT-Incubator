ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Should be set to either 'ubuntu' or 'rockylinux'
# depending on the BASE_IMAGE.
ARG LINUX_DISTRO=notset

#############################
# Basic Dependencies
#############################

ARG DEBIAN_FRONTEND=noninteractive

RUN <<EOF
set -e
case "${LINUX_DISTRO}" in
  "ubuntu"*)
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
    ;;
  "rockylinux"*)
    # This script mostly comes from RAPIDS manylinux-compatible rockylinux
    # container build script.
    dnf update -y
    dnf install -y epel-release
    dnf update -y
    dnf install -y \
      which wget gcc zlib-devel bzip2 bzip2-devel readline-devel sqlite \
      sqlite-devel xz xz-devel libffi-devel curl git ncurses-devel \
      openssh-clients zip jq \
      protobuf-compiler autoconf automake libtool dnf-plugins-core cmake
    dnf config-manager --set-enabled powertools
    dnf -y install gcc-toolset-11-gcc gcc-toolset-11-gcc-c++
    dnf -y install ninja-build ccache
    dnf clean all
    # Enable GCC11 toolset as default.
    echo -e ' \
      #!/bin/bash\n \
      source /opt/rh/gcc-toolset-11/enable \
    ' > /etc/profile.d/enable_devtools.sh
    # Build and install openssl 1.1. This is required to build modern Python
    # versions. Paths from this install are passed to pyenv when building Python (see below
    # Python setup section).
    pushd tmp
    wget https://www.openssl.org/source/openssl-1.1.1k.tar.gz
        tar -xzvf openssl-1.1.1k.tar.gz
    cd openssl-1.1.1k
    ./config --prefix=/usr --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic
    make
    make install
    popd
    ;;
  *)
    echo "Unsupported linux distro: ${LINUX_DISTRO}"
    exit 1
    ;;
esac
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
COPY integrations/python/requirements-dev.txt /tmp/requirements-dev.txt
COPY integrations/python/requirements.txt /tmp/requirements.txt
RUN <<EOF
set -e
case "${LINUX_DISTRO}" in
  "ubuntu"*)
    python3 -m pip install -r /tmp/requirements-dev.txt
    ;;
  "rockylinux"*)
    curl https://pyenv.run | bash
    pyenv update
    CPPFLAGS="-I/usr/include/openssl" LDFLAGS="-L/usr/lib" pyenv install --verbose 3.9 3.10 3.11 3.12
    pyenv global ${PYTHON_VERSION}
    chmod -R 0755 /pyenv
    ;;
  *)
    echo "Unsupported LINUX_DISTRO: ${LINUX_DISTRO}"
    exit 1
    ;;
esac
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
case "${LINUX_DISTRO}" in
  "ubuntu"*)
    chmod +x /tmp/install_tools.sh
    /tmp/install_tools.sh
    ;;
  "rockylinux"*)
    echo "Skipping LLVM install for rocklinux"
    ;;
  *)
    echo "Unsupported LINUX_DISTRO: ${LINUX_DISTRO}"
    exit 1
    ;;
esac
rm /tmp/install_tools.sh
EOF
