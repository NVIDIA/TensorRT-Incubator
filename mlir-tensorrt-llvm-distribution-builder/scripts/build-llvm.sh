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

# Install Python requirements (always install for clean builds, check for incremental)
if [[ "${INCREMENTAL_BUILD}" == "true" ]] && python -c "import nanobind" 2>/dev/null; then
  echo "==> Skipping Python requirements installation (already installed)"
else
  echo "==> Installing Python requirements..."
  python -m pip install -r $LLVM_PROJECT_DIR/mlir/python/requirements.txt
fi

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


# copy the mlir pdll include files to the install directory
if [[ ! -d "${LLVM_INSTALL_DIR}/share/mlir/include" || -z "$(ls -A "${LLVM_INSTALL_DIR}/share/mlir/include" 2>/dev/null)" ]]; then
  mkdir -p "${LLVM_INSTALL_DIR}/share/mlir/include"
  cp -r ${LLVM_PROJECT_DIR}/mlir/include/* \
    "${LLVM_INSTALL_DIR}/share/mlir/include/"
  echo "==> Successfully copied mlir include files to ${LLVM_INSTALL_DIR}/share/mlir/include"
else
  echo "==> Skipping copy: ${LLVM_INSTALL_DIR}/share/mlir/include already exists and is non-empty"
fi

# add the llvm-lit script to use the lit from python package lit (skip if it already exists)
if [[ ! -x "${LLVM_INSTALL_DIR}/bin/llvm-lit" ]]; then
  cat > "${LLVM_INSTALL_DIR}/bin/llvm-lit" <<'PYWRAP'
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
  chmod +x "${LLVM_INSTALL_DIR}/bin/llvm-lit"
  echo "==> Successfully added llvm-lit wrapper to ${LLVM_INSTALL_DIR}/bin/llvm-lit"
else
  echo "==> Skipping llvm-lit wrapper: ${LLVM_INSTALL_DIR}/bin/llvm-lit already exists"
fi

echo "==> LLVM build completed successfully"