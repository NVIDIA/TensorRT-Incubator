#!/usr/bin/env bash
# Builds Python wheels for MLIR-TensorRT packages using the specified python version number.
# Usage: build_wheels.sh 3.X
set -e
version=${1:-3.10}

mkdir -p .private.wheels || true

pyenv local ${version}

python${version} -m pip install -r python/requirements-dev.txt

rm -rf build || true
cmake -B ./build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_TRT_PACKAGE_CACHE_DIR=${PWD}/.cache.cpm \
    -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON -DLLVM_MINIMUM_PYTHON_VERSION=${version} \
    -DMLIR_TRT_ENABLE_ASSERTIONS=OFF \
    -DMLIR_TRT_ENABLE_NCCL=OFF \
    -DMLIR_TRT_DOWNLOAD_TENSORRT_VERSION=10.2

ninja -C build mlir-tensorrt-all-wheels

rsync -za build/wheels/ .private.wheels/
