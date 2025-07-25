#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -ex

docker build -f ${SCRIPT_DIR}/Dockerfile \
             -t mlir-tensorrt:dev \
             --build-arg GROUP=$(id -gn) \
             --build-arg GID=$(id -g) \
             --build-arg USER=$(id -un) \
             --build-arg UID=$(id -u) \
             .

docker run -it \
           -v "${SCRIPT_DIR}/../":"/opt/src/mlir-tensorrt" \
           --gpus all \
           mlir-tensorrt:dev
