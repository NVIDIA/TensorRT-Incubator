#!/usr/bin/env bash
set -eo pipefail

# This script is for installing the recommended development tools.
# Updated to support both Ubuntu and Rocky Linux.

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LLVM_VERSION="${LLVM_VERSION:-18}"
# TODO: if you update the version, you still need to update the sha256 hashes below.
UV_VERSION="${UV_VERSION:-0.9.13}"
echo "Installing developer tools and LLVM ${LLVM_VERSION} toolchain."

# Function to detect the distribution
detect_distro() {
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "$ID"
  else
    echo "unknown"
  fi
}

DISTRO=$(detect_distro)
echo "Detected distribution: $DISTRO"

arch=$(uname -p)

compare_hash() {
  local file_path
  file_path="$1"
  local expected_hash
  expected_hash="$2"

  if [[ ! -f "$file_path" ]]; then
    echo "Error: File '$file_path' does not exist."
    exit 1
  fi

  local actual_hash
  actual_hash=$(sha256sum "$file_path" | awk '{print $1}')

  if [[ "$actual_hash" != "$expected_hash" ]]; then
    echo "Error: SHA256 hash mismatch for '$file_path'."
    echo "Expected: $expected_hash"
    echo "Actual:   $actual_hash"
    exit 1
  fi
}

function curl_with_retry() {
  local curl_url="$1"
  local curl_output="${2:-}"
  local curl_expected_hash="${3:-}"
  curl -L --retry 5 --retry-delay 5 --retry-connrefused --silent \
    -o "$curl_output" "$curl_url"
  compare_hash "$curl_output" "$curl_expected_hash"
}

function install_cmake() {
  local url
  local expected_hash
  if [[ ${arch} == "x86_64" ]]; then
    url="https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2-linux-x86_64.sh"
    expected_hash="0bdecd361a8bc22e91122372cf9ec83711ca786d14aadee6988001189b151b96"
  elif [[ ${arch} == "aarch64" ]]; then
    url="https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2-linux-aarch64.sh"
    expected_hash="05ea676cfd5ed5b13f2f2c25a30a470cdb6270817812e1f6505e3d48dd7defd2"
  else
    echo "Warning: Upgraded 'CMake' is unavailable for arch=${arch}"
    return 0
  fi
  curl_with_retry "$url" "install-cmake.sh" \
    "$expected_hash"
  chmod +x install-cmake.sh
  ./install-cmake.sh --prefix=/usr/local/ --skip-license
  rm install-cmake.sh
}

function install_uv() {
  # URLS and hashes from https://github.com/astral-sh/uv/releases
  if [[ ${arch} == "x86_64" ]]; then
    curl_with_retry "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
      uv.tar.gz \
      c45a44144bf23a2182e143227b4ad0bbe41a2bb7161a637c02e968906af53fd1
  elif [[ ${arch} == "aarch64" ]]; then
    curl_with_retry "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-aarch64-unknown-linux-gnu.tar.gz" \
      uv.tar.gz \
      c221d04810f873a7aa8bae9aa6ed721e600e56534980df1363952386a4fcdcc5
  else
    echo "Warning: 'uv' is unavailable for arch=${arch}"
    return 1
  fi
  tar --strip-components=1 -xzf uv.tar.gz
  rm uv.tar.gz
  mv uv /usr/local/bin/uv
  mv uvx /usr/local/bin/uvx
}

# Distribution-specific package installation
case "$DISTRO" in
"ubuntu" | "debian")
  echo "Installing packages for Ubuntu/Debian..."
  apt-get update -qq

  # Install CMake and LLVM debian packages from their public debian repositories.
  ${SCRIPT_DIR}/install-llvm-debian.sh ${LLVM_VERSION}

  DEV_TOOLS="ccache unzip git git-lfs mold sudo"
  GCC_TOOLSET="gcc g++"
  DEV_LIBS="zlib1g-dev bzip2 libbz2-dev libreadline-dev libsndfile1"
  PYTHON_PACKAGES="python3 python3-dev python3-venv python3-pip"
  OPENMPI_PACKAGES="openmpi-bin openmpi-common libopenmpi-dev"
  apt-get install -y --no-install-recommends -qq \
   ${DEV_TOOLS} ${GCC_TOOLSET} ${DEV_LIBS} ${PYTHON_PACKAGES} ${OPENMPI_PACKAGES}

  # Use update-alternatives to set the default LLVM toolchain.
  ${SCRIPT_DIR}/update-llvm-toolchain-alternatives.sh ${LLVM_VERSION} 100

  # Clean up
  apt-get clean -y -qq
  rm -rf /var/lib/apt/lists/*
  ;;
"rocky" | "centos" | "rhel" | "almalinux")
  echo "Installing packages for Rocky Linux/RHEL..."

  # Install EPEL repository if not already installed
  if ! rpm -q epel-release >/dev/null 2>&1; then
    dnf install -yq epel-release
  fi

  # Enable PowerTools/CRB repository for additional development packages
  if [ "$DISTRO" = "rocky" ] && [ "$(rpm -E %{rhel})" = "8" ]; then
    dnf config-manager --set-enabled powertools || dnf config-manager --set-enabled PowerTools || true
  elif [ "$DISTRO" = "rocky" ] && [ "$(rpm -E %{rhel})" = "9" ]; then
    dnf config-manager --set-enabled crb || true
  fi

  install_cmake

  DEV_TOOLS="git git-lfs ccache unzip mold sudo"
  if [ "$DISTRO" = "rocky" ] && [ "$(rpm -E %{rhel})" = "8" ]; then
    GCC_TOOLSET="gcc-toolset-11-gcc gcc-toolset-11-gcc-c++"
  else
    GCC_TOOLSET="gcc gcc-c++"
  fi
  DEV_LIBS="zlib-devel bzip2 bzip2-devel readline-devel libsndfile"
  # Note: No libc++ package available in EPEL.
  LLVM_PACKAGES_ROCKY="clang lld lldb clang-tools-extra git-clang-format"
  PYTHON_PACKAGES="python3.12 python3.12-devel python3.12-pip"
  OPENMPI_PACKAGES="openmpi openmpi-devel"

  # Install basic development tools and LLVM packages
  # Note: Rocky Linux repositories typically have specific LLVM versions available
  dnf install -yq ${DEV_TOOLS} \
    ${GCC_TOOLSET} \
    ${DEV_LIBS} \
    ${PYTHON_PACKAGES} ${OPENMPI_PACKAGES} \
    ${LLVM_PACKAGES_ROCKY}
  ;;
*)
  echo "Error: Unsupported distribution: $DISTRO"
  echo "Supported distributions: ubuntu, debian, rocky, centos, rhel, almalinux"
  exit 1
  ;;
esac

# Install uv
install_uv

# Install CMake
install_cmake

# Install newer version of Ninja (distribution-agnostic)
install_ninja() {
  if command -v ninja >/dev/null 2>&1; then
    echo "Error: 'ninja' already exists at $(command -v ninja). Aborting installation to avoid overwriting." >&2
    exit 1
  fi
  curl_with_retry "$1" ninja.zip "$2"
  unzip ninja.zip
  rm ninja.zip
  mv ninja /usr/local/bin/ninja
}

if [[ ${arch} == "x86_64" ]]; then
  install_ninja "https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip" \
    "6f98805688d19672bd699fbbfa2c2cf0fc054ac3df1f0e6a47664d963d530255"
elif [[ ${arch} == "aarch64" ]]; then
  install_ninja "https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux-aarch64.zip" \
    "5c25c6570b0155e95fce5918cb95f1ad9870df5768653afe128db822301a05a1"
else
  echo "Warning: Upgraded 'Ninja' is unavailable for arch=${arch}; skipping"
fi


echo "Done!"
