#!/usr/bin/env bash
# MLIR-TensorRT CI/CD Build Wheels Script
#
# This script builds Python wheels for MLIR-TensorRT packages.
#
# USAGE:
#     ./build_tools/scripts/cicd-build-wheels.sh
#
# DESCRIPTION:
#     This script builds Python wheels for MLIR-TensorRT packages.
#
# Environment Variables (overridable; defaults provided):
#     WHEELS_DIR - Directory to store built wheels (default: ${REPO_ROOT}/dist).
#                  It can be a relative or absolute path; relative paths are resolved against REPO_ROOT.
#     PYTHON_VERSIONS / python_versions - Space or comma separated Python versions to build
#                  (default: "3.10 3.11 3.12 3.13"; examples: "3.10", "3.10 3.11", "3.11,3.12")
#     PACKAGES - Space or comma separated list of packages to build wheels for
#                  (default: "mlir_tensorrt_tools mlir_tensorrt_compiler mlir_tensorrt_runtime")
#                  (examples: "mlir_tensorrt_tools", "mlir_tensorrt_compiler,mlir_tensorrt_runtime")
#     MLIR_TRT_DOWNLOAD_TENSORRT_VERSION - The TensorRT version to build the wheels for (default: 10.13)
#     MLIR_TRT_BUILD_DIR - The directory to build the project in (default: ${REPO_ROOT}/build)
#     MLIR_TRT_INSTALL_DIR - The directory to install the project in (default: ${REPO_ROOT}/install)
#     VERBOSE - Enable verbose output (default: 0, set to 1 for verbose)
#
# EXAMPLES:
#     # Basic usage: build wheels for all packages and all supported Python versions
#     ./build_tools/scripts/cicd-build-wheels.sh
#
#     # Build wheels with verbose output
#     VERBOSE=1 ./build_tools/scripts/cicd-build-wheels.sh
#
#     # Build wheel for a single package and a single Python version
#     PACKAGES=mlir_tensorrt_runtime PYTHON_VERSIONS="3.10" ./build_tools/scripts/cicd-build-wheels.sh
#
#     # Build wheels for multiple python versions and multiple packages
#     PACKAGES="mlir_tensorrt_tools mlir_tensorrt_compiler" PYTHON_VERSIONS="3.10 3.11" ./build_tools/scripts/cicd-build-wheels.sh
#

set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found in PATH. Please install uv (https://astral.sh/uv) before running this script." >&2
  exit 1
fi

SCRIPT_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export MLIR_TRT_BUILD_DIR="${MLIR_TRT_BUILD_DIR:-${REPO_ROOT}/build}"
mkdir -p "${MLIR_TRT_BUILD_DIR}"
export MLIR_TRT_INSTALL_DIR="${MLIR_TRT_INSTALL_DIR:-${REPO_ROOT}/install}"
mkdir -p "${MLIR_TRT_INSTALL_DIR}"
export WHEELS_DIR="${WHEELS_DIR:-${REPO_ROOT}/dist}"

export MLIR_TRT_DOWNLOAD_TENSORRT_VERSION="${MLIR_TRT_DOWNLOAD_TENSORRT_VERSION:-10.13}"

# Resolve Python versions
_raw_py_versions="${PYTHON_VERSIONS:-${python_versions:-"3.10 3.11 3.12 3.13"}}"
_raw_py_versions="${_raw_py_versions//,/ }"
read -r -a PY_VERSIONS <<< "${_raw_py_versions}"

# Resolve packages: if PACKAGES is set, only build those; otherwise build all defaults
if [[ -n "${PACKAGES:-}" ]]; then
  _raw_packages="${PACKAGES}"
else
  # default: build all available packages
  _raw_packages="mlir_tensorrt_compiler mlir_tensorrt_runtime pjrt"
fi
_raw_packages="${_raw_packages//,/ }"
read -r -a PACKAGES <<< "${_raw_packages}"

function build_package() {
  local pkg=$1
  for pyver in "${PY_VERSIONS[@]}"; do
    ccache --zero-stats || true
    time -p uv build --wheel --out-dir="${WHEELS_DIR}" --python="${pyver}" "${pkg}"
    ccache --show-stats || true
  done
}

# Ensure auditwheel exists
function ensure_auditwheel_installed() {
  if ! command -v auditwheel >/dev/null 2>&1; then
    echo "auditwheel not found; installing into the system environment..."
    uv pip install --system "auditwheel>=6.0.0"
  fi
}

# Ensure patchelf exists
function ensure_patchelf_installed() {
  if ! command -v patchelf >/dev/null 2>&1; then
    echo "patchelf not found; attempting to install a recent patchelf..."
    mkdir patchelf
    pushd .
    cd patchelf
    curl -fsSL -o patchelf.tar.gz "https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-$(uname -m).tar.gz"
    tar -xzf patchelf.tar.gz
    cp bin/patchelf /usr/bin/patchelf || true
    chmod +x /usr/bin/patchelf || true
    popd
    rm -rf patchelf
    echo "patchelf installed: $(patchelf --version 2>/dev/null || echo 'unknown')"
  fi
}

# Perform auditwheel repair (computes default platform by arch, respects AUDITWHEEL_PLAT)
function auditwheel_repair() {
  local arch default_plat repair_plat
  arch="$(uname -m)"
  case "${arch}" in
    x86_64)  default_plat="manylinux_2_28_x86_64" ;;
    aarch64) default_plat="manylinux_2_28_aarch64" ;;
    *)       default_plat="" ;;
  esac
  repair_plat="${AUDITWHEEL_PLAT:-${default_plat}}"
  if [[ -z "${repair_plat}" ]]; then
    echo "Unknown architecture '${arch}', skipping auditwheel repair."
    return 0
  fi
  echo "Repairing wheels in '${WHEELS_DIR}' to '${repair_plat}' if needed..."
  shopt -s nullglob
  for whl in "${WHEELS_DIR}"/*.whl; do
    if [[ "${whl}" == *linux_x86_64.whl || "${whl}" == *linux_aarch64.whl ]]; then
      echo "Listing the contents of the wheels before repair"
      unzip -l "${whl}"

      echo "Running auditwheel repair on ${whl}"
      cat ${REPO_ROOT}/build_tools/scripts/soname_excludes.params
      auditwheel repair $(cat ${REPO_ROOT}/build_tools/scripts/soname_excludes.params) --plat "${repair_plat}" -w "${WHEELS_DIR}" "${whl}" || {
        echo "WARNING: auditwheel failed to repair ${whl} for ${repair_plat}. The wheel may not be uploadable to PyPI." >&2
        continue
      }

      rm -f "${whl}"

      echo "Listing the contents of the wheels after repair"
      prefix="${whl%-linux_*.whl}"
      repaired_whl="${prefix}-manylinux*.whl"
      unzip -l "${repaired_whl}"

    fi
  done
  shopt -u nullglob
}

# Map package keys to paths and build
for pkg_key in "${PACKAGES[@]}"; do
  case "${pkg_key}" in
    mlir_tensorrt_compiler)
      build_package "integrations/python/mlir_tensorrt_compiler"
      ;;
    mlir_tensorrt_runtime)
      build_package "integrations/python/mlir_tensorrt_runtime"
      ;;
    pjrt|PJRT)
      build_package "integrations/PJRT/python"
      ;;
    *)
      echo "WARNING: Unknown package key '${pkg_key}'. Supported: mlir_tensorrt_compiler, mlir_tensorrt_runtime, pjrt" >&2
      ;;
  esac
done

if [[ "${MLIR_TRT_PYPI:-0}" == "1" ]]; then
  echo "🔨🧪 auditwheel repair for PyPI upload."
  # Attempt to repair Linux wheels to a manylinux-compliant tag so PyPI/TestPyPI accepts them.
  if [[ "$(uname -s)" == "Linux" ]]; then
    ensure_auditwheel_installed
    ensure_patchelf_installed
    auditwheel_repair
  fi
fi



# print the size of the ccache and cpm source cache
echo "🔨🧪 Printing the size of the ccache"
du -h -x -d 1 ${CCACHE_DIR}
echo "🔨🧪 Printing the size of the cpm source cache"
du -h -x -d 1 ${CPM_SOURCE_CACHE}

echo "🔨🧪 Printing the size of the wheels directory"
du -h -x -d 1 "${WHEELS_DIR}"

ls -lart "${WHEELS_DIR}"
