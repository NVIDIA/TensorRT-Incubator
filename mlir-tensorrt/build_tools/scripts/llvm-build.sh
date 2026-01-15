#!/usr/bin/env bash
#
# Build LLVM package from a local clone
# Usage: ./llvm-build.sh --target-dir <dir>
# Options:
#   --target-dir <dir>   Directory to clone into (default: third_party/llvm-project)


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default target directory
TARGET_DIR="${REPO_ROOT}/third_party/llvm-project"
INSTALL_DIR="${REPO_ROOT}/third_party/llvm-install-dir"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--target-dir <dir>] [--install-dir <dir>]"
      echo ""
      echo "Options:"
      echo "  --target-dir <dir>   Directory to clone into (default: third_party/llvm-project)"
      echo "  --install-dir <dir>  Directory to install LLVM-Project (default: third_party/llvm-install-dir)"
      echo ""
      echo "This script Build LLVM-Project from a local clone for MLIR development."
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--target-dir <dir>]"
      exit 1
      ;;
  esac
done

echo "==> Target directory: ${TARGET_DIR}"
echo "==> Install directory: ${INSTALL_DIR}"


uv venv -p 3.12 --clear ./venv_312
source venv_312/bin/activate
arch="$(uname -m)"

pushd .
cd third_party

# Generate build
cmake -GNinja -Bllvm-project/build \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
-DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
-DCMAKE_LINKER=lld \
-DLLVM_BUILD_UTILS=ON \
-DLLVM_BUILD_TOOLS=ON \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
-DLLVM_ENABLE_PROJECTS="mlir" \
-DLLVM_INSTALL_UTILS=ON \
-DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
-DLLVM_ENABLE_TERMINFO=OFF \
-DLLVM_ENABLE_ZSTD=OFF \
llvm-project/llvm

# Install
ninja -C llvm-project/build install

popd


