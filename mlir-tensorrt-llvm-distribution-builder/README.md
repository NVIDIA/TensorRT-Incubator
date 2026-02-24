This repo contains configuration and scripts to build a distribution of LLVM and MLIR development binaries, headers, and CMake configuration files for use with MLIR-TRT development.

These are used to create packages which are posted to the internal gitlab registry so that they can be used by MLIR-TRT developers instead of having to build LLVM from source.

## Configuration

Relevant configuration is kept in `config.json`.

## Building

```shell
# Install dependencies
pixi install
# start pixi shell environment
pixi shell

# Clones llvm-project to `./llvm-project` subdirectory and checks out the correct branch
./scripts/setup-llvm.sh

# Builds llvm-project
export CMAKE_BUILD_TYPE=Release
./scripts/build-llvm.sh

# Tar the install directory and upload
tar -czf llvm-project-$(uname)-$(uname -m) -C install/$CMAKE_BUILD_TYPE .

# TODO: upload to Gitlab package registry (as a generic package).

```
