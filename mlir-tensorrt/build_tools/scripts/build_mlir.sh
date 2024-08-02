#!/usr/bin/env bash
# usage: build_mlir.sh [src dir] [build dir]
set -e

src_dir="${1:-llvm-project}"
build_dir="${2:-build/llvm-project}"
cmake -GNinja \
  "-S${src_dir}/llvm" \
  "-B${build_dir}" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_LLD="$ON" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_VERSION_SUFFIX="" \
  -DMLIR_INCLUDE_TESTS=ON \
  -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_CCACHE_BUILD=ON

ninja -C "${build_dir}" all
