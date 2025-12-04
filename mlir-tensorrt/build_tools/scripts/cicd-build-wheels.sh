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
#     WHEELS_DIR - Directory to store built wheels (default: ${REPO_ROOT}/.wheels).
#                  It can be a relative or absolute path; relative paths are resolved against REPO_ROOT.
#     PYTHON_VERSIONS / python_versions - Space or comma separated Python versions to build
#                  (default: "3.10 3.11 3.12 3.13"; examples: "3.10", "3.10 3.11", "3.11,3.12")
#     PACKAGES - Space or comma separated list of packages to build wheels for
#                  (default: "mlir_tensorrt_tools mlir_tensorrt_compiler mlir_tensorrt_runtime")
#                  (examples: "mlir_tensorrt_tools", "mlir_tensorrt_compiler,mlir_tensorrt_runtime")
#     MTRT_TENSORRT_VERSION - The TensorRT version to build the wheels for (default: 10.13)
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
export WHEELS_DIR="${WHEELS_DIR:-${REPO_ROOT}/.wheels}"
mkdir -p "${WHEELS_DIR}"

export MTRT_TENSORRT_VERSION="${MTRT_TENSORRT_VERSION:-10.13}"

# Resolve Python versions
_raw_py_versions="${PYTHON_VERSIONS:-${python_versions:-"3.10 3.11 3.12 3.13"}}"
_raw_py_versions="${_raw_py_versions//,/ }"
read -r -a PY_VERSIONS <<< "${_raw_py_versions}"

# Resolve packages
_raw_packages="${PACKAGES:-"mlir_tensorrt_tools mlir_tensorrt_compiler mlir_tensorrt_runtime"}"
_raw_packages="${_raw_packages//,/ }"
read -r -a PACKAGES <<< "${_raw_packages}"

cd "${REPO_ROOT}"
for pkg in "${PACKAGES[@]}"; do
  for pyver in "${PY_VERSIONS[@]}"; do
    ccache --zero-stats || true
    time -p uv build --wheel --out-dir="${WHEELS_DIR}" --python="${pyver}" "integrations/python/${pkg}"
    ccache --show-stats || true
  done
done

# print the size of the ccache and cpm source cache
echo "🔨🧪 Printing the size of the ccache"
du -h -x -d 1 ${CCACHE_DIR}
echo "🔨🧪 Printing the size of the cpm source cache"
du -h -x -d 1 ${CPM_SOURCE_CACHE}

echo "🔨🧪 Printing the size of the wheels directory"
du -h -x -d 1 "${WHEELS_DIR}"

ls -lart "${WHEELS_DIR}"