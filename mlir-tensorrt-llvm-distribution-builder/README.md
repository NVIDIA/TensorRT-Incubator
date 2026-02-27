This repo contains configuration and scripts to build a distribution of LLVM and MLIR development binaries, headers, and CMake configuration files for use with MLIR-TRT development.

These are used to create packages which are posted to the internal gitlab registry so that they can be used by MLIR-TRT developers instead of having to build LLVM from source.

## Configuration

Relevant configuration is kept in `config.json`.

## Clone llvm-project

```shell
# Clones llvm-project to `./llvm-project` subdirectory and checks out the correct branch
cd TensorRT-Incubator
./mlir-tenosrrt/build_tools/scripts/setup-llvm-dev.sh \
--target-dir ./mlir-tensorrt-llvm-distribution-builder/llvm-project
```

## Building

```shell
# install pixi if it it not installed
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

# Install dependencies
pixi install
# start pixi shell environment
pixi shell

# Builds llvm-project
export CMAKE_BUILD_TYPE=Release
./scripts/build-llvm.sh

# Tar the install directory and upload
tar -czf llvm-project-$(uname)-$(uname -m) -C install/$CMAKE_BUILD_TYPE .


```
## Verify the prebuilt llvm package

```shell

```