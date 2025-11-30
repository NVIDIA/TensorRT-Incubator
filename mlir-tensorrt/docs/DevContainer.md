# Using the Development Container

We provide several pre-configured development containers for use.

## Prerequisites

The following software is required:

1. [Docker](https://docs.docker.com/engine/install/ubuntu/)
2. [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

For VSCode integration, you will also need the [Microsoft's Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.

## Variants

We provide several different containers based on Linux distro (Ubuntu & RockyLinux variatns) and
CUDA version.

**Variants with the `-prebuilt` suffix will attempt to use a pre-built image that we build and push to [our GitHub registry](https://github.com/NVIDIA/TensorRT-Incubator/pkgs/container/tensorrt-incubator%2Fmlir-tensorrt)**.

If for some reason the pre-built variants do not work, the variants without the `-prebuilt` tag will cause
your system to try to build the container from scratch, which can take several minutes.

## Usage

We assume you have a recent version of VSCode installed with the Dev Containers extension.

1. Start VS Code and open your cloned repository in VSCode. VSCode will prompt you to launch the container. Or you can run the **Dev Containers: Reopen Folder in Container...** command from the Command Palette (F1) or quick actions Status bar item.
2. We provide several different container configurations. VSCode will prompt you to select one.

Prefer use of the `*-prebuilt` containers, as that will save a lot of time.
