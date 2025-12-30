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
set -ex
case "${LINUX_DISTRO}" in
  "ubuntu"*)
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
        wget curl
    apt-get clean -y -qq
    rm -rf /var/lib/apt/lists/*
    ;;
  "rockylinux"*)
    # Lock CUDA so it doesn't get upgraded.
    dnf install -yq python3-dnf-plugin-versionlock
    dnf versionlock add '*cuda*'
    dnf versionlock add '*cudnn*'
    dnf install -yq wget patch
    ;;
  *)
    echo "Unsupported linux distro: ${LINUX_DISTRO}"
    exit 1
    ;;
esac
EOF

#------------------------------------------------------------------------------
# Install developer tools (e.g. LLVM toolchain, CMake, etc.)
#------------------------------------------------------------------------------
ARG LLVM_VERSION=20
ENV LLVM_VERSION=$LLVM_VERSION
COPY build_tools/docker/*.sh /tmp/scripts/
RUN /tmp/scripts/install-toolchain.sh && rm -rf /tmp/scripts
