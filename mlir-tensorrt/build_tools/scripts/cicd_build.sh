#!/usr/bin/env bash
set -ex
set -o pipefail

REPO_ROOT=$(pwd)
BUILD_DIR="${REPO_ROOT}/build"
LLVM_LIT_ARGS=${LLVM_LIT_ARGS:-"-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --timeout=1200 --time-tests -Drun_long_tests=${RUN_LONG_TESTS}"}
DOWNLOAD_TENSORRT_VERSION=${DOWNLOAD_TENSORRT_VERSION:-10.5}
ENABLE_ASAN=${ENABLE_ASAN:-OFF}

export CCACHE_DIR="/ccache"

ccache --zero-stats || true
rm -rf ${BUILD_DIR}  || true

echo "Using DOWNLOAD_TENSORRT_VERSION=${DOWNLOAD_TENSORRT_VERSION}"
echo "Using LLVM_LIT_ARGS=${LLVM_LIT_ARGS}"

cmake --preset ninja-llvm \
  -DMLIR_TRT_DOWNLOAD_TENSORRT_VERSION="${DOWNLOAD_TENSORRT_VERSION}" \
  -DENABLE_ASAN="${ENABLE_ASAN}" \
  -DLLVM_LIT_ARGS="${LLVM_LIT_ARGS}" \
  -DCPM_SOURCE_CACHE="/.cache.cpm" \
  -DMLIR_EXECUTOR_ENABLE_GPU_INTEGRATION_TESTS=OFF

ninja -C ${BUILD_DIR} -k 0 check-all-mlir-tensorrt
ccache --show-stats || true
