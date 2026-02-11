#!/usr/bin/env bash
#===----------------------------------------------------------------------===//
# Setup a local LLVM-Project clone for MLIR-TensorRT development.
#
# This script clones llvm-project with sparse checkout to minimize disk usage,
# checks out the required commit, and applies MLIR-TensorRT patches.
#
# Usage:
#   ./build_tools/scripts/setup-llvm-dev.sh [--target-dir <dir>]
#
# Options:
#   --target-dir <dir>   Directory to clone into (default: third_party/llvm-project)
#
# After running this script, add the following to your CMakeUserPresets.json:
#   "CPM_LLVM_SOURCE": "${sourceDir}/third_party/llvm-project"
#===----------------------------------------------------------------------===//
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PATCH_DIR="${REPO_ROOT}/build_tools/patches/mlir"

# Default target directory
TARGET_DIR="${REPO_ROOT}/third_party/llvm-project"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--target-dir <dir>]"
      echo ""
      echo "Options:"
      echo "  --target-dir <dir>   Directory to clone into (default: third_party/llvm-project)"
      echo ""
      echo "This script sets up a local LLVM-Project clone for MLIR development."
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--target-dir <dir>]"
      exit 1
      ;;
  esac
done

# Extract LLVM commit from DependencyProvider.cmake
LLVM_COMMIT=$(sed -n 's/.*MLIR_TRT_LLVM_COMMIT[[:space:]]*"\([^"]*\)".*/\1/p' "${REPO_ROOT}/DependencyProvider.cmake")
if [[ -z "${LLVM_COMMIT}" ]]; then
  echo "Error: Could not extract MLIR_TRT_LLVM_COMMIT from DependencyProvider.cmake"
  exit 1
fi

echo "==> LLVM commit: ${LLVM_COMMIT}"
echo "==> Target directory: ${TARGET_DIR}"

# Count available patches
PATCHES=($(ls "${PATCH_DIR}"/*.patch 2>/dev/null | sort))
NUM_PATCHES=${#PATCHES[@]}

# Function to apply patches
apply_patches() {
  echo "==> Applying MLIR-TensorRT patches..."
  if [[ ${NUM_PATCHES} -eq 0 ]]; then
    echo "    No patches found in ${PATCH_DIR}"
  else
    for patch in "${PATCHES[@]}"; do
      patch_name=$(basename "${patch}")
      echo "    Applying: ${patch_name}"
      git am "${patch}"
    done
    echo "    Applied ${NUM_PATCHES} patches."
  fi
}

# Check if directory already exists
if [[ -d "${TARGET_DIR}" ]]; then
  echo "==> Directory ${TARGET_DIR} already exists."
  echo "    Checking current state..."

  cd "${TARGET_DIR}"
  CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")

  if [[ "${CURRENT_COMMIT}" == "${LLVM_COMMIT}" ]]; then
    echo "    At base LLVM commit. Patches need to be applied."
    apply_patches
    echo ""
    echo "==> Setup complete!"
    exit 0
  fi

  # Check if we're at LLVM_COMMIT + patches (i.e., NUM_PATCHES commits ahead)
  if git merge-base --is-ancestor "${LLVM_COMMIT}" HEAD 2>/dev/null; then
    COMMITS_AHEAD=$(git rev-list "${LLVM_COMMIT}..HEAD" --count)
    if [[ "${COMMITS_AHEAD}" -eq "${NUM_PATCHES}" ]]; then
      echo "    Already at LLVM commit with ${NUM_PATCHES} patches applied."
      echo "==> Setup complete!"
      exit 0
    else
      echo "    Found ${COMMITS_AHEAD} commits ahead of base LLVM commit (expected ${NUM_PATCHES} patches)."
      echo "    The repository may have additional local changes."
      echo "==> Assuming setup is complete."
      exit 0
    fi
  else
    echo "Error: Directory exists but base commit ${LLVM_COMMIT} is not an ancestor of HEAD."
    echo "       Current HEAD: ${CURRENT_COMMIT}"
    echo "       Please remove the directory or use a different --target-dir."
    exit 1
  fi
fi

# Create parent directory if needed
mkdir -p "$(dirname "${TARGET_DIR}")"

echo "==> Cloning llvm-project with sparse checkout..."
echo "    This minimizes download size and improves IDE performance."

# Clone with blob filtering to reduce initial download size
git clone --filter=blob:none --no-checkout \
  https://github.com/llvm/llvm-project.git \
  "${TARGET_DIR}"

cd "${TARGET_DIR}"

# Initialize sparse checkout
echo "==> Initializing sparse checkout..."
git sparse-checkout init --cone

# Set the directories we need for MLIR development
# - cmake: CMake modules
# - llvm: LLVM core (required)
# - mlir: MLIR dialect and passes
# - third-party: Dependencies like googlebenchmark
# - utils: Utilities like lit, TableGen backends
echo "==> Setting sparse checkout paths: cmake llvm mlir third-party utils"
git sparse-checkout set cmake llvm mlir third-party utils

# Checkout the required commit
echo "==> Checking out commit ${LLVM_COMMIT}..."
git checkout "${LLVM_COMMIT}"

git config --global user.email "lanl@nvidia.com"
git config --global user.name "Lan Luo"
# Apply patches
apply_patches

echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add the following to your CMakeUserPresets.json:"
echo ""
echo '     {'
echo '       "version": 6,'
echo '       "configurePresets": ['
echo '         {'
echo '           "name": "my-config",'
echo '           "inherits": "default",'
echo '           "cacheVariables": {'
echo '             "CPM_LLVM_SOURCE": "${sourceDir}/third_party/llvm-project"'
echo '           }'
echo '         }'
echo '       ]'
echo '     }'
echo ""
echo "  2. Reconfigure your build:"
echo "     cmake --preset my-config --fresh"
echo ""
