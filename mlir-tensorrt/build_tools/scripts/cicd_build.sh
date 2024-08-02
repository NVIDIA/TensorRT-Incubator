#!/usr/bin/env bash
set -ex
set -o pipefail

REPO_ROOT=$(pwd)
BUILD_DIR="${BUILD_DIR:=${REPO_ROOT}/build/mlir-tensorrt}"

ENABLE_NCCL=${ENABLE_NCCL:OFF}
RUN_LONG_TESTS=${RUN_LONG_TESTS:-False}
LLVM_LIT_ARGS=${LLVM_LIT_ARGS:-"-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --timeout=1200 --time-tests -Drun_long_tests=${RUN_LONG_TESTS}"}
DOWNLOAD_TENSORRT_VERSION=${DOWNLOAD_TENSORRT_VERSION:-10.0.0.6}
ENABLE_ASAN=${ENABLE_ASAN:-OFF}

echo "Using DOWNLOAD_TENSORRT_VERSION=${DOWNLOAD_TENSORRT_VERSION}"
echo "Using LLVM_LIT_ARGS=${LLVM_LIT_ARGS}"

cmake -GNinja -B "${BUILD_DIR}" -S "${REPO_ROOT}" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_TRT_USE_LINKER=lld -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMLIR_TRT_PACKAGE_CACHE_DIR=$PWD/.cache.cpm \
  -DMLIR_TRT_ENABLE_PYTHON=ON \
  -DMLIR_TRT_ENABLE_NCCL=${ENABLE_NCCL} \
  -DMLIR_TRT_DOWNLOAD_TENSORRT_VERSION="$DOWNLOAD_TENSORRT_VERSION" \
  -DLLVM_LIT_ARGS="${LLVM_LIT_ARGS}" \
  -DENABLE_ASAN="${ENABLE_ASAN}" \
  -DMLIR_DIR=${REPO_ROOT}/build/llvm-project/lib/cmake/mlir \
  -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON

echo "==== Running Build ==="
ninja -C ${BUILD_DIR} -k 0 check-mlir-executor
ninja -C ${BUILD_DIR} -k 0 check-mlir-tensorrt
ninja -C ${BUILD_DIR} -k 0 check-mlir-tensorrt-dialect
