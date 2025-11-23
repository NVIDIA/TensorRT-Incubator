#!/usr/bin/env bash
# MLIR-TensorRT CI/CD Build and Test Script
#
# This script performs a complete build and test cycle for MLIR-TensorRT.
# It includes dependency installation, CMake configuration, building, and testing.
#
set -euo pipefail

# Help function
show_help() {
  cat <<'EOF'
MLIR-TensorRT CI/CD Build and Test Script

USAGE:
    cicd_build_test.sh

DESCRIPTION:
    This script performs a complete build and test cycle for MLIR-TensorRT.
    It includes dependency installation, CMake configuration, building, and testing.
    It also supports various presets for different build configurations.

ENVIRONMENT VARIABLES:
    CMAKE_PRESET       - CMake preset to use (default: github-cicd)
    REPO_ROOT          - Repository root directory (mlir-tensorrt root)
    BUILD_DIR          - Build directory (default: ${REPO_ROOT}/build)
    DOWNLOAD_TENSORRT_VERSION - TensorRT version to download (default: 10.12)
    CPM_SOURCE_CACHE   - CPM source cache directory (default: ${REPO_ROOT}/.cache.cpm)
    CCACHE_DIR         - CCache directory (default: ${REPO_ROOT}/ccache)
    SKIP_TESTS         - Skip tests (default: 0, set to 1 to skip)
    VERBOSE            - Enable verbose output (default: 0, set to 1 for verbose)

EXAMPLES:
    # Basic usage: build and test all
    ./cicd_build_test.sh

    # Verbose output, build only, skip tests
    SKIP_TESTS=1 VERBOSE=1 ./cicd_build_test.sh

    # build and test with a custom preset (e.g., AddressSanitizer build):
    CMAKE_PRESET=github-cicd-with-asan ./cicd_build_test.sh
    # build and test with NCCL and long tests
    CMAKE_PRESET=github-cicd-with-nccl-long-tests ./cicd_build_test.sh

AVAILABLE CMAKE PRESETS:
    Run 'cmake --list-presets=configure' from the repository root to see all available presets.
    Common presets include:


EXIT CODES:
    0   - Success
    1   - General error (missing dependencies, configuration failure, build failure)
    2   - Invalid arguments or usage

For more information, see the project documentation.
EOF
}

# Handle help requests
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

# Enable verbose output if requested
if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BUILD_DIR=${BUILD_DIR:-${REPO_ROOT}/build}

# clean up the build directory if it exists
rm -rf "${BUILD_DIR}" || true

export LLVM_LIT_ARGS=${LLVM_LIT_ARGS:-"-v --xunit-xml-output ${BUILD_DIR}/test-results.xml --timeout=1200 --time-tests -Drun_long_tests=${LONG_TESTS:-false}"}
export DOWNLOAD_TENSORRT_VERSION=${DOWNLOAD_TENSORRT_VERSION:-10.12}
export CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-${REPO_ROOT}/.cache.cpm}
export CCACHE_DIR=${CCACHE_DIR:-${REPO_ROOT}/ccache}
export SKIP_TESTS=${SKIP_TESTS:-0}
export VERBOSE=${VERBOSE:-0}
export CMAKE_PRESET=${CMAKE_PRESET:-github-cicd}
export NINJA_JOBS="${NINJA_JOBS:-$(nproc 2>/dev/null || echo 4)}"


# function for starting the section
function section_start() {
  local section_title="${1}"
  local section_description="${2:-$section_title}"

  echo -e "section_start:$(date +%s):${section_title}[collapsed=true]\r\e[0K${section_description}"
}

# Function for ending the section
function section_end() {
  local section_title="${1}"

  echo -e "section_end:$(date +%s):${section_title}\r\e[0K"
}

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
  echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $*" >&2
}

# Error handling
cleanup_on_error() {
  local exit_code=$?
  log_error "Script failed with exit code: $exit_code"
  if [[ -n "${BUILD_DIR:-}" && -d "$BUILD_DIR" ]]; then
    log_info "Build directory: $BUILD_DIR"
    log_info "Recent build logs may be available in the build directory"
  fi
  exit $exit_code
}

trap cleanup_on_error ERR


# Validate required commands
check_command() {
  if ! command -v "$1" &>/dev/null; then
    log_error "Required command '$1' not found. Please install it and try again."
    exit 1
  fi
}

install_python_dependencies() {
  local requirements_file="${REPO_ROOT}/pyproject.toml"
  if [[ ! -f "$requirements_file" ]]; then
    log_error "pyproject.toml not found: $requirements_file"
    return 1
  fi
  log_info "Installing Python dependencies..."
  if [[ -d "${REPO_ROOT}/.venv" ]]; then
    rm -rf "${REPO_ROOT}/.venv"
  fi
  uv venv "${REPO_ROOT}/.venv" || { log_error "uv venv failed under ${REPO_ROOT}"; return 1; }
  source "${REPO_ROOT}/.venv/bin/activate" || { log_error "failed to activate virtual environment"; return 1; }
  local frozen_flag=""
  if [[ -f "uv.lock" ]]; then
    frozen_flag="--frozen"
    log_info "Using uv.lock for reproducible installs"
  fi
  pushd "${REPO_ROOT}" >/dev/null
  if [[ -n "${CUDA_VERSION:-}" ]]; then
     cuda_major=$(echo "$CUDA_VERSION" | cut -d. -f1)
     log_info "Detected CUDA_VERSION=${CUDA_VERSION}; installing extra: cu${cuda_major}"
     uv sync --no-install-project ${frozen_flag} --extra "cu${cuda_major}" || { log_error "uv sync failed for extra cu${cuda_major}"; deactivate >/dev/null 2>&1 || true; popd >/dev/null; return 1; }
  else
     log_info "CUDA_VERSION not set; installing extra: cpu"
    uv sync --no-install-project ${frozen_flag} --extra "cpu" || { log_error "uv sync failed for extra cpu"; deactivate >/dev/null 2>&1 || true; popd >/dev/null; return 1; }
  fi
  log_info "Python dependencies installed successfully"
  popd >/dev/null
  return 0
}

configure_cmake() {
  log_info "Configuring CMake with preset: $CMAKE_PRESET"
  section_start "cmake_configure"
  # Show CMake version for debugging
  cmake --version
  if [[ -d "$BUILD_DIR" ]]; then
    log_info "Cleaning existing build directory: $BUILD_DIR ..."
    rm -rf "$BUILD_DIR"
  fi
  mkdir -p "${BUILD_DIR}"
  if ! cmake -B "$BUILD_DIR" -S "$REPO_ROOT" --preset "$CMAKE_PRESET"; then
    log_error "CMake configuration failed"
    section_end "cmake_configure"
    return 1
  fi
  log_success "CMake configuration completed successfully"
  section_end "cmake_configure"
  return 0
}

# Show build statistics
show_build_stats() {
  log_info "Build statistics:"
  # ccache stats
  if command -v ccache &>/dev/null; then
    log_info "ccache statistics:"
    ccache --show-stats
  fi
  # Build directory size
  if [[ -d "$BUILD_DIR" ]]; then
    local build_size
    build_size=$(du -sh "$BUILD_DIR" 2>/dev/null | cut -f1 || echo "unknown")
    log_info "Build directory size: $build_size"
  fi
  # System resource usage
  if command -v free &>/dev/null; then
    log_info "Memory usage:"
    free -h
  fi
  return 0
}

# Main execution with comprehensive error handling
main() {
    log_info "Starting MLIR-TensorRT CI/CD Build and Test..."
    log_info "Configuration:"
    log_info "  Repository root: $REPO_ROOT"
    log_info "  Build directory: $BUILD_DIR"
    log_info "  CMAKE_PRESET: $CMAKE_PRESET"
    log_info "  Skip tests: $SKIP_TESTS"
    log_info "  VERBOSE: $VERBOSE"
    log_info "  LLVM_LIT_ARGS: $LLVM_LIT_ARGS"
    log_info "  DOWNLOAD_TENSORRT_VERSION: $DOWNLOAD_TENSORRT_VERSION"
    log_info "  CPM_SOURCE_CACHE: $CPM_SOURCE_CACHE"
    log_info "  CCACHE_DIR: $CCACHE_DIR"


    log_info "Checking required dependencies..."
    check_command uv
    check_command ccache
    check_command cmake
    check_command ninja
    check_command python3

    if ! install_python_dependencies; then
    log_error "❌ Failed to install Python dependencies"
    exit 1
    fi

    if ! configure_cmake; then
    log_error "❌ Failed to configure CMake"
    exit 1
    fi

    ccache --zero-stats || true

    if [[ "${SKIP_TESTS}" == "1" ]]; then
    log_info "🔨 Building only (skipping tests)..."
    ninja -C "${BUILD_DIR}" -k 0 -j "${NINJA_JOBS}" all
    else
    log_info "🔨🧪 Building and testing..."
    ninja -C "${BUILD_DIR}" -k 0 -j "${NINJA_JOBS}" check-all-mlir-tensorrt
    fi

    show_build_stats
    return 0
}

# Run the main function
main "$@"