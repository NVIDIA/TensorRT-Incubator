# Contributing to MLIR-TensorRT

## Using the Development Container

Currently, we support building MLIR-TensorRT using [VSCode dev-container](https://code.visualstudio.com/docs/devcontainers/containers) integration.

### Prerequisites

The following software is required:

1. [Docker](https://docs.docker.com/engine/install/ubuntu/)
2. [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

### Build and launch container 
We assume you have a recent version of VSCode installed with the Dev Containers extension.

1. Start VS Code and open your cloned repository in VSCode. VSCode will prompt you to launch the container. Or you can run the **Dev Containers: Reopen Folder in Container...** command from the Command Palette (F1) or quick actions Status bar item.
2. VSCode will build the development container and mount in the repository for your ease of use.

## Building MLIR-TensorRT from source

Follow(../mlir-tensorrt/README.md# Building) for building MLIR-TensorRT library and python bindings.
