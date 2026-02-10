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
LONG_TESTS=${LONG_TESTS:-False}
CMAKE_PRESET=${CMAKE_PRESET:-"github-cicd"}
export LLVM_LIT_ARGS=${LLVM_LIT_ARGS:-"-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --timeout=1200 --time-tests -Drun_long_tests=${LONG_TESTS}"}
export MLIR_TRT_DOWNLOAD_TENSORRT_VERSION=${MLIR_TRT_DOWNLOAD_TENSORRT_VERSION:-10.13}
export CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-${REPO_ROOT}/.cache.cpm}
export CCACHE_DIR=${CCACHE_DIR:-${REPO_ROOT}/ccache}

# Choose uv extra based on CUDA_VERSION major (12 -> cu12, 13 -> cu13).
# Fallback to probing nvcc if CUDA_VERSION is not set. Default to cu12.
UV_EXTRA="cu12"
if [[ -n "${CUDA_VERSION:-}" ]]; then
  CUDA_MAJOR="${CUDA_VERSION%%.*}"
  if [[ "${CUDA_MAJOR}" == "13" ]]; then
      UV_EXTRA="cu13"
  elif [[ "${CUDA_MAJOR}" == "12" ]]; then
      UV_EXTRA="cu12"
  else
      echo "CUDA_VERSION: ${CUDA_VERSION} is not supported"
      exit 1
  fi
else
  echo "CUDA_VERSION is not set, using default of cu12"
fi

echo "🔨 Syncing with uv extra: ${UV_EXTRA}"
uv sync --extra "${UV_EXTRA}"
source .venv/bin/activate

ccache --zero-stats || true
rm -rf ${BUILD_DIR}  || true

function build_with_preset() {
  local preset_name
  preset_name=$1
  cmake -B "${BUILD_DIR}" --preset "${preset_name}" --fresh
  echo "🔨 Building with preset: ${preset_name}"
  if [[ "$BUILD_ONLY" == "true" ]]; then
    echo "🔨 Building only (skipping tests)..."
    ninja -C ${BUILD_DIR} -k 0 all
  else
    echo "🔨🧪 Building and testing..."
    ninja -C ${BUILD_DIR} -k 0 check-all-mlir-tensorrt
  fi
}

build_with_preset ${CMAKE_PRESET}

ccache --show-stats || true

# print the size of the ccache and cpm source cache
echo "🔨🧪 Printing the size of the ccache"
du -h -x -d 1 ${CCACHE_DIR}
echo "🔨🧪 Printing the size of the cpm source cache"
du -h -x -d 1 ${CPM_SOURCE_CACHE}
