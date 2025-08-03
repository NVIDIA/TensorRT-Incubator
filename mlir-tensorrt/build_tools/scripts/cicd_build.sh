#!/usr/bin/env bash
set -ex
set -o pipefail

# Parse command line arguments
BUILD_ONLY=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --build_only)
      BUILD_ONLY=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--build_only]"
      echo "  --build_only: Only build, skip tests"
      exit 1
      ;;
  esac
done

REPO_ROOT=$(pwd)
BUILD_DIR="${REPO_ROOT}/build"
export LLVM_LIT_ARGS=${LLVM_LIT_ARGS:-"-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --timeout=1200 --time-tests -Drun_long_tests=${RUN_LONG_TESTS}"}
export DOWNLOAD_TENSORRT_VERSION=${DOWNLOAD_TENSORRT_VERSION:-10.9}
export ENABLE_ASAN=${ENABLE_ASAN:-OFF}
export CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-/.cache.cpm}
export CCACHE_DIR=${CCACHE_DIR:-/ccache}

python3 -m pip install -r integrations/python/requirements-dev.txt --use-pep517

ccache --zero-stats || true
rm -rf ${BUILD_DIR}  || true

cmake -B${BUILD_DIR} --preset github-cicd

if [[ "$BUILD_ONLY" == "true" ]]; then
  echo "🔨 Building only (skipping tests)..."
  ninja -C ${BUILD_DIR} -k 0 all
else
  echo "🔨🧪 Building and testing..."
  ninja -C ${BUILD_DIR} -k 0 check-all-mlir-tensorrt
fi

ccache --show-stats || true
