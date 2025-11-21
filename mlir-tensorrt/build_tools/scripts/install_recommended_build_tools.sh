#!/usr/bin/env bash
set -e
# This script is for installing the recommended development tools (Clang, CMake,
# Ninja, ccache, etc) in a Debian-like environment. Must be run as root.
LLVM_VERSION="${LLVM_VERSION:-16}"
echo "Installing updated CMake, Ninja, ccache, and Clang/LLVM ${LLVM_VERSION} toolchain."

# Install CMake, Ninja, and LLVM, and other dev tools.
MLIR_TRT_BUILD_DEPS="cmake ccache unzip"
# The LLVM packages we should install for the compilation toolchain (e.g. clang, lld, etc).
LLVM_PACKAGES="clang-${LLVM_VERSION} lldb-${LLVM_VERSION} lld-${LLVM_VERSION} clangd-${LLVM_VERSION} clang-tools-${LLVM_VERSION} llvm-${LLVM_VERSION}-tools clang-format-${LLVM_VERSION}"

# Install CMake and LLVM debian package repositories.
source /etc/os-release
UBUNTU_CODENAME_DETECTED="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"
if [[ -z "${UBUNTU_CODENAME_DETECTED}" ]]; then
    echo "Warning: Could not detect Ubuntu codename; defaulting to 'jammy'."
    UBUNTU_CODENAME_DETECTED="jammy"
fi
echo "Using Ubuntu codename: ${UBUNTU_CODENAME_DETECTED}"
echo "deb http://apt.llvm.org/${UBUNTU_CODENAME_DETECTED}/ llvm-toolchain-${UBUNTU_CODENAME_DETECTED}-${LLVM_VERSION} main" > /etc/apt/sources.list.d/llvm.list
echo "deb-src http://apt.llvm.org/${UBUNTU_CODENAME_DETECTED}/ llvm-toolchain-${UBUNTU_CODENAME_DETECTED}-${LLVM_VERSION} main" >> /etc/apt/sources.list.d/llvm.list
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key 2>/dev/null > /etc/apt/trusted.gpg.d/apt.llvm.org.asc
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME_DETECTED} main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null
apt-get update
apt-get install -y ${LLVM_PACKAGES} ${MLIR_TRT_BUILD_DEPS}
apt-get clean -y
rm -rf /var/lib/apt/lists/*

# Sets up proper ubuntu/debian alternative symlinks for llvm packages. See this
# comment in this old gist:
# https://gist.github.com/junkdog/70231d6953592cd6f27def59fe19e50d?permalink_comment_id=4336074#gistcomment-4336074
update_alternatives() {
    local version=${1}
    local priority=${2}
    local master=${3}
    local slaves=${4}
    local path=${5}
    local cmdln
    cmdln="--verbose --install ${path}${master} ${master} ${path}${master}-${version} ${priority}"
    for slave in ${slaves}; do
        cmdln="${cmdln} --slave ${path}${slave} ${slave} ${path}${slave}-${version}"
    done
    update-alternatives ${cmdln}
}

version=${LLVM_VERSION}
priority=10
path="/usr/bin/"
master="llvm-config"
slaves="llvm-addr2line llvm-ar llvm-as llvm-bcanalyzer llvm-bitcode-strip llvm-cat llvm-cfi-verify llvm-cov llvm-c-test llvm-cvtres llvm-cxxdump llvm-cxxfilt llvm-cxxmap llvm-debuginfod llvm-debuginfod-find llvm-diff llvm-dis llvm-dlltool llvm-dwarfdump llvm-dwarfutil llvm-dwp llvm-exegesis llvm-extract llvm-gsymutil llvm-ifs llvm-install-name-tool llvm-jitlink llvm-jitlink-executor llvm-lib llvm-libtool-darwin llvm-link llvm-lipo llvm-lto llvm-lto2 llvm-mc llvm-mca llvm-ml llvm-modextract llvm-mt llvm-nm llvm-objcopy llvm-objdump llvm-omp-device-info llvm-opt-report llvm-otool llvm-pdbutil llvm-PerfectShuffle llvm-profdata llvm-profgen llvm-ranlib llvm-rc llvm-readelf llvm-readobj llvm-reduce llvm-remark-size-diff llvm-rtdyld llvm-sim llvm-size llvm-split llvm-stress llvm-strings llvm-strip llvm-symbolizer llvm-tapi-diff llvm-tblgen llvm-tli-checker llvm-undname llvm-windres llvm-xray"
update_alternatives "${version}" "${priority}" "${master}" "${slaves}" "${path}"
master="clang"
slaves="analyze-build asan_symbolize bugpoint c-index-test clang++ clang-apply-replacements clang-change-namespace clang-check clang-cl clang-cpp clangd clang-doc clang-extdef-mapping clang-format clang-format-diff clang-include-fixer clang-linker-wrapper clang-move clang-nvlink-wrapper clang-offload-bundler clang-offload-packager clang-offload-wrapper clang-pseudo clang-query clang-refactor clang-rename clang-reorder-fields clang-repl clang-scan-deps clang-tidy count diagtool dsymutil FileCheck find-all-symbols git-clang-format hmaptool hwasan_symbolize intercept-build ld64.lld ld.lld llc lld lldb lldb-argdumper lldb-instr lldb-server lldb-vscode lld-link lli lli-child-target modularize not obj2yaml opt pp-trace run-clang-tidy sancov sanstats scan-build scan-build-py scan-view split-file UnicodeNameMappingGenerator verify-uselistorder wasm-ld yaml2obj yaml-bench"
update_alternatives "${version}" "${priority}" "${master}" "${slaves}" "${path}"

arch=$(uname -p)

# Install ripgrep.
if [[ ${arch} == "x86_64" ]]; then
    wget https://github.com/BurntSushi/ripgrep/releases/download/13.0.0/ripgrep_13.0.0_amd64.deb \
        -O /tmp/ripgrep.deb
    dpkg -i /tmp/ripgrep.deb
    rm /tmp/ripgrep.deb
else
    echo "Warning: ripgrep is unavailable for arch=${arch}; skipping"
fi

# Newer version of Ninja
install_ninja() {
    wget ${url} -O ninja.zip
    unzip ninja.zip
    rm /usr/bin/ninja || true
    rm /usr/local/bin/ninja || true
    rm ninja.zip
    mv ninja /usr/local/bin/ninja
}
if [[ ${arch} == "x86_64" ]]; then
    url="https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip"
    install_ninja
elif [[ ${arch} == "aarch64" ]]; then
    url="https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux-aarch64.zip"
    install_ninja
else
    echo "Warning: Upgraded 'Ninja' is unavailable for arch=${arch}; skipping"
fi

echo "Done!"