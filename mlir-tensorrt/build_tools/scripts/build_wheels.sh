#!/usr/bin/env bash
# Builds Python wheels for MLIR-TensorRT packages using the specified python version number.
# Usage:
# build_wheels.sh [python version] [TensorRT version]
# e.g. build_wheels.sh 3.X 10.X
set -e
py_version=${PY_VERSION:-${1:-3.10}}

PATH=$PATH:/pyenv/bin
mkdir -p .private.wheels || true
pyenv local ${py_version}
python${py_version} -m pip install -r python/requirements-dev.txt

# If `DOWNLOAD_TENSORRT_VERSION` is not set, use the second argument as the version number.
if [ -z "${DOWNLOAD_TENSORRT_VERSION}" ]; then
    DOWNLOAD_TENSORRT_VERSION=$2
    export DOWNLOAD_TENSORRT_VERSION
fi

rm -rf build || true
cmake --preset ninja-clang-wheel-release
ninja -C build mlir-tensorrt-all-wheels
rsync -za build/wheels/ .private.wheels/
