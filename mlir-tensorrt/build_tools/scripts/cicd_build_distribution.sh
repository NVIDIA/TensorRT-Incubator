#!/usr/bin/env bash
# MLIR-TensorRT CI/CD Build Distribution Script
#
# This script builds the distribution tar.gz for MLIR-TensorRT.
#
# USAGE:
#     ./build_tools/scripts/cicd_build_distribution.sh
#
# DESCRIPTION:
#     This script builds the distribution tar.gz for MLIR-TensorRT.
#
# Environment Variables (overridable; defaults provided):
#     PKG_DIR                   - Directory to place the generated tar.gz (default: ${REPO_ROOT}/install)
#     PKG_FILE                  - Output tarball filename; must end with .tar.gz
#                                 (default: mlir-tensorrt-$(uname -m)-linux-tensorrt${DOWNLOAD_TENSORRT_VERSION}.tar.gz)
#     DOWNLOAD_TENSORRT_VERSION - TensorRT version to download (default: 10.12)
#                                 If unset, will fall back to TENSORRT_VERSION, else 10.12.
#     TENSORRT_VERSION          - Optional fallback if DOWNLOAD_TENSORRT_VERSION is not set.
#     CMAKE_PRESET              - CMake preset to use (default: distribution)
#     SKIP_TESTS                - Skip tests (default: 1; set to 0 to run tests before building the distribution)
#     VERBOSE                   - Enable verbose output (default: 0; set to 1 for verbose)
#
# EXAMPLES:
#     # Basic usage: build distribution tar.gz using all the default values
#     ./build_tools/scripts/cicd_build_distribution.sh
#
#     # Verbose output, build distribution tar.gz with specific TensorRT version
#     PKG_FILE=mlir-tensorrt-x86_64-linux-cuda13.0-tensorrt10.13.tar.gz \
#     DOWNLOAD_TENSORRT_VERSION=10.13 \
#     VERBOSE=1 \
#     ./build_tools/scripts/cicd_build_distribution.sh
#
set -euo pipefail


# Enable verbose output if requested
if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

# Establish repo root and defaults
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export DOWNLOAD_TENSORRT_VERSION="${DOWNLOAD_TENSORRT_VERSION:-${TENSORRT_VERSION:-10.12}}"
export CMAKE_PRESET="${CMAKE_PRESET:-distribution}"
export SKIP_TESTS="${SKIP_TESTS:-1}"
host_arch="$(uname -m)"
export PKG_DIR="${PKG_DIR:-${REPO_ROOT}/install}"
default_pkg_name="mlir-tensorrt-${host_arch}-linux-tensorrt${DOWNLOAD_TENSORRT_VERSION}.tar.gz"
export PKG_FILE="${PKG_FILE:-${default_pkg_name}}"
mkdir -p "${PKG_DIR}"


if [[ "${PKG_FILE}" != *.tar.gz ]]; then
  echo "PKG_FILE must end with .tar.gz"
  exit 1
fi

folder_name="$(basename "${PKG_FILE}" .tar.gz)"
export BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
export MLIR_TRT_INSTALL_DIR="${PKG_DIR}/${folder_name}/mlir-tensorrt"

# --- Helper to parse Version.cmake ---
parse_cmake_var() {
  local var="$1"
  sed -nE "s/^[[:space:]]*set[[:space:]]*\\([[:space:]]*$var[[:space:]]+\"?([0-9]+)\"?[[:space:]]*\\).*/\\1/p" "${REPO_ROOT}/Version.cmake"
}

MAJOR=$(parse_cmake_var "MLIR_TENSORRT_VERSION_MAJOR" || true)
MINOR=$(parse_cmake_var "MLIR_TENSORRT_VERSION_MINOR" || true)
PATCH=$(parse_cmake_var "MLIR_TENSORRT_VERSION_PATCH" || true)

if [[ -z "$MAJOR" || -z "$MINOR" || -z "$PATCH" ]]; then
  echo "❌ ERROR: Could not parse version from Version.cmake"
  exit 1
fi

TS=$(date -u +%Y%m%d%H%M%S)
VERSION="${MAJOR}.${MINOR}.${PATCH}.dev${TS}"
TAG="v${VERSION}"

echo "→ Version: ${VERSION}"
echo "→ Tag: ${TAG}"
echo "→ Install directory: ${MLIR_TRT_INSTALL_DIR}"

${SCRIPT_DIR}/cicd_build_test.sh


# clean up the install directory if it exists
rm -rf "${MLIR_TRT_INSTALL_DIR}"
mkdir -p "${MLIR_TRT_INSTALL_DIR}"
# TODO: ASK Chris whether we can copy packaging.cmake from gitlab to github
# ninja -C build install-mtrt-distribution

echo "→ Creating archive for ${folder_name}..."

# Determine output path (respect absolute/relative PKG_FILE)
if [[ "${PKG_FILE}" == */* ]]; then
  PKG_OUT="${PKG_FILE}"
else
  PKG_OUT="${PKG_DIR}/${PKG_FILE}"
fi

if command -v pigz &>/dev/null; then
  echo "  Using pigz for compression"
  tar -cf "${PKG_OUT}" -I pigz -C "${PKG_DIR}" "${folder_name}"
else
  echo "  Using gzip for compression"
  tar -czf "${PKG_OUT}" -C "${PKG_DIR}" "${folder_name}"
fi

echo "  ✅ Created ${PKG_OUT}"


