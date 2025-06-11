#!/usr/bin/env bash
# Builds Python wheels for MLIR-TensorRT packages using the specified python version number.
# e.g.
# PY_VERSION=3.10 build_wheels.sh
# PY_VERSION=3.10 DOWNLOAD_TENSORRT_VERSION=10.9 build_wheels.sh
set -e
py_version=${PY_VERSION:-3.10}

PATH=$PATH:/pyenv/bin
mkdir -p .private.wheels || true
pyenv local ${py_version}
python${py_version} -m pip install -r integrations/python/requirements-dev.txt

export DOWNLOAD_TENSORRT_VERSION=${DOWNLOAD_TENSORRT_VERSION:-10.9}

rm -rf build || true
cmake --preset ninja-release-wheels
ninja -C build mlir-tensorrt-all-wheels
rsync -za build/wheels/ .private.wheels/
