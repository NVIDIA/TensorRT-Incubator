#!/usr/bin/env bash
#
# Build LLVM package from a local clone
# Usage: ./build-llvm.sh [--target-dir <dir>] [--install-dir <dir>] [--incremental-build]
# Options:
#   --target-dir <dir>      Directory which you have cloned llvm-project into (default: $PWD/llvm-project)
#   --install-dir <dir>     Directory to install LLVM-Project (default: $PWD/install/${CMAKE_BUILD_TYPE})
#   --incremental-build     Use incremental build (reuse existing build). By default, builds from scratch.
#
# This script Build LLVM-Project from a local clone for MLIR development.
set -euo pipefail


CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
INCREMENTAL_BUILD=false

LLVM_PROJECT_DIR="${LLVM_PROJECT_DIR:-$PWD/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$PWD/build/${CMAKE_BUILD_TYPE}}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-$PWD/install/${CMAKE_BUILD_TYPE}}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --target-dir)
      LLVM_PROJECT_DIR="$2"
      shift 2
      ;;
    --install-dir)
      LLVM_INSTALL_DIR="$2/${CMAKE_BUILD_TYPE}"
      shift 2
      ;;
    --incremental-build)
      INCREMENTAL_BUILD=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--target-dir <dir>] [--install-dir <dir>] [--incremental-build]"
      echo ""
      echo "Options:"
      echo "  --target-dir <dir>      Directory which you have cloned llvm-project into (default: third_party/llvm-project)"
      echo "  --install-dir <dir>     Directory to install LLVM-Project (default: third_party/llvm-install-dir)"
      echo "  --incremental-build     Use incremental build (reuse existing build). By default, builds from scratch."
      echo ""
      echo "This script Build LLVM-Project from a local clone for MLIR development."
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--target-dir <dir>] [--install-dir <dir>] [--incremental-build]"
      exit 1
      ;;
  esac
done

echo "==> LLVM project directory: ${LLVM_PROJECT_DIR}"
echo "==> LLVM build directory: ${LLVM_BUILD_DIR}"
echo "==> LLVM install directory: ${LLVM_INSTALL_DIR}"

if [[ "${INCREMENTAL_BUILD}" == "true" ]]; then
  echo "==> Incremental build requested - reusing existing build directory if present"
  if [[ -d "${LLVM_BUILD_DIR}" ]]; then
    echo "    Found existing build directory: ${LLVM_BUILD_DIR}"
  else
    echo "    No existing build directory - will configure from scratch"
  fi
else
  echo "==> Clean build from scratch - removing existing build and install directories"
  rm -rf $LLVM_BUILD_DIR || true
  rm -rf $LLVM_INSTALL_DIR || true
fi

mkdir -p $LLVM_BUILD_DIR
mkdir -p $LLVM_INSTALL_DIR

# Python dependencies are managed by pixi via requirements.txt
# Install them using: LLVM_PROJECT_DIR=${LLVM_PROJECT_DIR} pixi run install-mlir-deps
# The build script assumes dependencies are already installed in the pixi environment

arch="$(uname -m)"

# Build cmake command with conditional --fresh flag
CMAKE_ARGS=(
  -C $PWD/cmake/caches/LLVMBuildConfig.cmake
  -S $LLVM_PROJECT_DIR/llvm
  -B $LLVM_BUILD_DIR
  -GNinja
  --toolchain $PWD/cmake/toolchains/linux-${arch}.cmake
  -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
  -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR
)

# Only skip --fresh for incremental builds
if [[ "${INCREMENTAL_BUILD}" == "true" ]]; then
  echo "==> Configuring CMake (incremental build - reusing cache)"
else
  CMAKE_ARGS+=(--fresh)
  echo "==> Configuring CMake with --fresh flag (clean build from scratch)"
fi

cmake "${CMAKE_ARGS[@]}"

echo "==> Building and installing LLVM..."
ninja -C $LLVM_BUILD_DIR install-distribution

du -h -x -d 1 ${LLVM_INSTALL_DIR}
du -h -x -d 1 ${LLVM_BUILD_DIR}
echo "==> LLVM build completed successfully"