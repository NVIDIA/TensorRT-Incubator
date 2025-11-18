#!/usr/bin/env bash
# e.g.
# DOWNLOAD_TENSORRT_VERSION=10.12 release_wheels.sh
set -e
set -x

REPO_ROOT=$(pwd)
BUILD_DIR="${REPO_ROOT}/build"
export CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-/.cache.cpm}
export CCACHE_DIR=${CCACHE_DIR:-/ccache}
# either DOWNLOAD_TENSORRT_VERSION or MLIR_TRT_TENSORRT_DIR must be set
# MLIR_TRT_TENSORRT_DIR must be a valid path to a local TensorRT installation
# DOWNLOAD_TENSORRT_VERSION will attempt todownload a specified version of TensorRT if cannot find in the cache
export DOWNLOAD_TENSORRT_VERSION=${DOWNLOAD_TENSORRT_VERSION:-10.9}

mkdir -p .private.wheels 

python3 -m pip install -r integrations/python/requirements-dev.txt --use-pep517
# ccache --zero-stats || true
rm -rf ${BUILD_DIR} || true

cmake -B ${BUILD_DIR} --preset python-wheel-build 

