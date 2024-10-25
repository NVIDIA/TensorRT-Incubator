#!/usr/bin/env bash

docker build -f build_tools/bazel/Dockerfile \
             -t mlir-tensorrt:dev \
			 --build-arg BASE_IMAGE=nvcr.io/nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04 \
             --build-arg GROUP=$(id -gn) \
             --build-arg GID=$(id -g) \
             --build-arg USER=$(id -un) \
             --build-arg UID=$(id -u) \
             .

docker run -it \
           -v "$(pwd)":"/opt/src/mlir-tensorrt" \
           -v "${HOME}/.cache/bazel":"${HOME}/.cache/bazel" \
           mlir-tensorrt:dev
