#!/usr/bin/env bash
# e.g.
# cicd_build_wheels.sh
set -euo pipefail
set -x

PY_CMD=${PY_CMD:-python3}

if [ -f /etc/os-release ] && grep -qi 'rocky' /etc/os-release; then
  yum install -y epel-release
  yum install -y python3.12 python3.12-pip python3.12-devel clang lld git ccache patch

fi


$PY_CMD -m pip install -U pip uv build || true

REPO_ROOT="$(pwd)"
ARCH="${ARCH:-$(uname -m)}"
CHANNEL="${CHANNEL:-test}"
TENSORRT_VERSION="${TENSORRT_VERSION:-10.12}"
# Defaults computed after ARCH is known
DEFAULT_BUILD_DIR="${REPO_ROOT}/build"
# CUDA_VERSION is like 13.0.0, 12.9.1, etc.
# cuda_major_minor_version is like 13.0, 12.9, etc.
cuda_major_minor_version=${CUDA_VERSION%.*}
DEFAULT_INSTALL_DIR="${REPO_ROOT}/install/mlir-tensorrt-${ARCH}-linux-cuda${cuda_major_minor_version=${CUDA_VERSION%.*}
}-tensorrt${TENSORRT_VERSION}/mlir-tensorrt"
WHEELS_DIR="${REPO_ROOT}/.wheels"

export DOWNLOAD_TENSORRT_VERSION=${TENSORRT_VERSION}

export MLIR_TRT_BUILD_DIR=${MLIR_TRT_BUILD_DIR:-${DEFAULT_BUILD_DIR}}
mkdir -p ${MLIR_TRT_BUILD_DIR}
export MLIR_TRT_INSTALL_DIR=${MLIR_TRT_INSTALL_DIR:-${DEFAULT_INSTALL_DIR}}
mkdir -p ${MLIR_TRT_INSTALL_DIR}
export MTRT_SKIP_TESTS=${MTRT_SKIP_TESTS:-1}

if [ "${CHANNEL}" == "release" ]; then
  PY_VERSIONS=("3.10" "3.11" "3.12" "3.13")
else
  PY_VERSIONS=("3.10")
fi

PACKAGES=("mlir_tensorrt_tools" "mlir_tensorrt_compiler" "mlir_tensorrt_runtime")

for pkg in "${PACKAGES[@]}"; do
  for pyver in "${PY_VERSIONS[@]}"; do
    ${PY_CMD} -m uv build --wheel --out-dir="${WHEELS_DIR}" --python="${pyver}" "integrations/python/${pkg}"
  done
done
