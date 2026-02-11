#!/usr/bin/env bash
#
# Build LLVM package from a local clone
# Usage: ./llvm-build.sh --target-dir <dir>
# Options:
#   --target-dir <dir>   Directory to clone into (default: third_party/llvm-project)

set -x

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
uv pip install -r ${TARGET_DIR}/mlir/python/requirements.txt
arch="$(uname -m)"

# pushd .
# cd third_party

# Generate build
cmake -GNinja -Bthird_party/llvm-project/build \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
-DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
-DCMAKE_LINKER=lld \
-DLLVM_BUILD_UTILS=ON \
-DLLVM_INCLUDE_TESTS=ON \
-DLLVM_BUILD_TOOLS=ON \
-DLLVM_INSTALL_UTILS=ON \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DMLIR_ENABLE_BINDINGS_PYTHON=ON \
-DMLIR_INSTALL_AGGREGATE_OBJECTS=ON \
-DLLVM_ENABLE_PROJECTS="mlir" \
-DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
-DLLVM_ENABLE_TERMINFO=OFF \
-DLLVM_ENABLE_ZSTD=OFF \
-DLLVM_INSTALL_GTEST=ON \
third_party/llvm-project/llvm

# Install
ninja -C third_party/llvm-project/build install

# copy the mlir pdll include files to the install directory
if [[ ! -d "${INSTALL_DIR}/share/mlir/include" || -z "$(ls -A "${INSTALL_DIR}/share/mlir/include" 2>/dev/null)" ]]; then
  mkdir -p "${INSTALL_DIR}/share/mlir/include"
  cp -r third_party/llvm-project/mlir/include/* \
    "${INSTALL_DIR}/share/mlir/include/"
else
  echo "==> Skipping copy: ${INSTALL_DIR}/share/mlir/include already exists and is non-empty"
fi


# add the llvm-lit script to use the lit from python package lit
cat > "${INSTALL_DIR}/bin/llvm-lit" <<'PYWRAP'
#!/usr/bin/env python3
import sys, shutil, os

def main():
    # Prefer running the lit module directly if available in this Python
    try:
        from lit.main import main as lit_main
        sys.exit(lit_main())
    except Exception:
        pass
    # Fallback: find a lit executable on PATH and exec it
    lit_path = shutil.which("lit")
    if lit_path:
        os.execv(lit_path, [lit_path] + sys.argv[1:])
    sys.stderr.write("Error: lit not found. Install it or add it to PATH.\n")
    sys.exit(1)

if __name__ == "__main__":
    main()
PYWRAP

chmod +x "${INSTALL_DIR}/bin/llvm-lit"

#popd


