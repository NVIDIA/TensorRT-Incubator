#!/usr/bin/env bash
set -ex
set -o pipefail

REPO_ROOT=$(pwd)
BUILD_DIR="${REPO_ROOT}/build"
export LLVM_LIT_ARGS=${LLVM_LIT_ARGS:-"-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --timeout=1200 --time-tests -Drun_long_tests=${RUN_LONG_TESTS}"}
export DOWNLOAD_TENSORRT_VERSION=${DOWNLOAD_TENSORRT_VERSION:-10.9}
export ENABLE_ASAN=${ENABLE_ASAN:-OFF}
export CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-/.cache.cpm}
export CCACHE_DIR=${CCACHE_DIR:-/ccache}

python3 -m pip install -r integrations/python/requirements-dev.txt

ccache --zero-stats || true
rm -rf ${BUILD_DIR}  || true

cmake -B${BUILD_DIR} --preset github-cicd

ninja -C ${BUILD_DIR} -k 0 check-all-mlir-tensorrt
ccache --show-stats || true
