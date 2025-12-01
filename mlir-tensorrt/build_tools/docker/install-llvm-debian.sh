#!/usr/bin/env bash
#
# install-llvm-debian.sh
#
# Usage:
#   sudo bash install-llvm-debian.sh
#
# Optional environment variables:
#   LLVM_VERSION=19

set -euo pipefail

LLVM_VERSION="${LLVM_VERSION:-19}"

echo "=== Installing LLVM/Clang ${LLVM_VERSION} ==="

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script must be run as root (or with sudo)." >&2
  exit 1
fi

# Detect Ubuntu codename if available
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  UBUNTU_CODENAME="${UBUNTU_CODENAME:-noble}"
else
  UBUNTU_CODENAME="noble"
fi

ARCH="$(dpkg --print-architecture)"

echo "Ubuntu codename: ${UBUNTU_CODENAME}"
echo "Architecture: ${ARCH}"

install_base_packages() {
  echo ">>> Installing base packages..."
  apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg \
    software-properties-common
}

add_llvm_repo() {
  local keyring="/usr/share/keyrings/llvm-archive-keyring.gpg"
  local list_file="/etc/apt/sources.list.d/llvm-${LLVM_VERSION}.list"

  echo ">>> Adding LLVM APT repository..."

  mkdir -p "$(dirname "${keyring}")"

  # Download and convert the key
  wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
    | gpg --dearmor > "${keyring}"

  # Add versioned repo
  cat > "${list_file}" <<EOF
deb [arch=${ARCH} signed-by=${keyring}] http://apt.llvm.org/${UBUNTU_CODENAME}/ llvm-toolchain-${UBUNTU_CODENAME}-${LLVM_VERSION} main
EOF
}

install_llvm() {
  echo ">>> Installing LLVM/Clang ${LLVM_VERSION}..."
  apt-get update
  local TOOL_PKGS="clang-$LLVM_VERSION lldb-$LLVM_VERSION lld-$LLVM_VERSION clangd-$LLVM_VERSION clang-format-$LLVM_VERSION"
  local LIBCPP_PKGS="libc++-$LLVM_VERSION-dev libc++abi-$LLVM_VERSION-dev"
  apt-get install -y -qq --no-install-recommends \
   $TOOL_PKGS \
   $LIBCPP_PKGS
}

main() {
  install_base_packages
  add_llvm_repo
  install_llvm
  echo "=== LLVM/Clang ${LLVM_VERSION} installation complete ==="
}

main "$@"
