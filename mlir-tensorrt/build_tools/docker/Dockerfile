ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Should be set to either 'ubuntu' or 'rockylinux'
# depending on the BASE_IMAGE.
ARG LINUX_DISTRO=notset

#############################
# Basic Dependencies
#############################

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}
ARG TZ=America/Los_Angeles
ENV TZ=${TZ}

RUN <<EOF
set -ex
case "${LINUX_DISTRO}" in
  "ubuntu"*)

    ubuntu_version=$(. /etc/os-release; echo "$VERSION_ID")
    # CUDA_VERSION is like 13.0.0, 12.9.1, etc.
    # cuda_major_minor_version is like 13.0, 12.9, etc.
    cuda_major_minor_version=${CUDA_VERSION%.*}
    # cuda_major_version is like 13, 12, etc.
    cuda_major_version=${cuda_major_minor_version%.*}
    # cuda_minor_version is like 0, 9, etc.
    cuda_minor_version=${cuda_major_minor_version#*.}

    if [ "${cuda_major_version}" = "13" ]; then
        # For CUDA 13.0, the nsight package name is like cuda-nsight-systems-13-0
        nsight_pkg_name="cuda-nsight-systems-${cuda_major_version}-${cuda_minor_version}"
    else
        # For CUDA 12.9, the nsight package name is like cuda-nsight-systems-12.9
        nsight_pkg_name="cuda-nsight-systems-${cuda_major_version}.${cuda_minor_version}"
    fi

    if [ "${ubuntu_version}" = "24.04" ]; then
        apt-get update && \
        apt-get install -y --no-install-recommends tzdata software-properties-common && \
        ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
        dpkg-reconfigure -f noninteractive tzdata && \
        apt-get install -y python3.12 python3.12-dev python3.12-venv python3-pip
    else
        apt-get update && apt-get install -y \
        python3.10 python3.10-distutils python3.10-dev python3.10-venv python3-pip
    fi
    apt-get update && apt-get install -y \
        wget gnupg apt-utils curl \
        default-jdk-headless default-jre-headless \
        git libprotobuf-dev protobuf-compiler libsndfile1 \
        ${nsight_pkg_name} clang-format \
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
    dnf -y install ninja-build ccache patch lld clang clang-tools-extra python3.12 python3.12-devel python3.12-pip
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
#------------------------------------------------------------------------------

RUN <<EOF
set -ex
curl -fsSL https://astral.sh/uv/install.sh | sh
ln -sf /root/.local/bin/uv /usr/local/bin/uv
uv python install 3.10 3.11 3.12 3.13
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
