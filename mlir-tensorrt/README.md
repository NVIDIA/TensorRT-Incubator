# MLIR-TensorRT Bindings, StableHLO Conversions, and More

This code contains:

- A [MLIR](https://mlir.llvm.org/) [dialect](https://mlir.llvm.org/docs/LangRef/#dialects)
  that attempts  to precisely model the [TensorRT operator set](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/).
  It provides an MLIR dialect, static verification and type inference, optimizations,
  translations to TensorRT by invoking the TensorRT builder API (`nvinfer1::INetworkBuilder`),
  and translations to C++ calls to the builder API.
- Conversions from [stablehlo](https://github.com/openxla/stablehlo) to the TensorRT dialect.
- An example compiler infrastructure for building a compiler and runtime that offloads complex
  sub-programs to TensorRT, complete with support for (bounded) dynamic shapes and a
  Python interface.

Note that the TensorRT dialect is under the top-level 'tensorrt' folder and can be
built as an independent project in case the other features are not needed for your
use-case.

# Building

We currently support only building on Linux x86 systems.

We support building several different ways (only via CMake) depending on use-case.

In each case, the LLVM-Project version that we are currently aligned to is
given in `build_tools/cmake/LLVMCommit.cmake`.

Note that currently we provide an LLVM patch which essentially cherry-picks the
bug fixes from [this open MLIR PR](https://github.com/llvm/llvm-project/pull/91524).

1. Build as a Standalone Project with LLVM downloaded by CMake
2. Build as a Standalone Project with LLVM provided by User
3. Build as a sub-project of a larger build (e.g. `add_subdirectory`)
4. Build via LLVM-External-Projects mechanism

Here we only show how to do Option 1 and Option 2.

## Option 1: Build as a Standalone Project with LLVM downloaded by CMake

This is the simplest way to get started and incurs low download overhead
since we download LLVM-Project as a zip archive directly from GitHub
at our pinned commit.

```sh
# Note: we use clang and lld here. These are recommended.
# However, GNU toolchains will also work,
# clang toolchain is optional.
cmake -B ./build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMLIR_TRT_PACKAGE_CACHE_DIR=${PWD}/.cache.cpm \
    -DMLIR_TRT_ENABLE_ASSERTIONS=ON \
    -DMLIR_TRT_DOWNLOAD_TENSORRT_VERSION=10.2 \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_TRT_USE_LINKER=lld

# Example build commands:

# Build everything
ninja -C build all

# Build and run tests
ninja -C build/mlir-tensorrt check-mlir-executor
ninja -C build/mlir-tensorrt check-mlir-tensorrt-dialect
ninja -C build/mlir-tensorrt check-mlir-tensorrt

# Build wheels (output in `build/wheels`)
ninja -C build mlir-tensorrt-all-wheels
```

## Option 2: Build as a Standalone Project with LLVM provided by User

This is more complex but lets you "bring your own LLVM-Project" source
code or binary.


1. Build MLIR

```sh
# Clone llvm-project
git clone https://github.com/llvm/llvm-project.git llvm-project

# Checkout the right commit. Of course, you may try
# a newer commit or your own modified LLVM-Project.
cd llvm-project
git checkout $(cat ../build_tools/cmake/LLVMCommit.cmake | grep -Po '(?<=").*(?=")')

# Apply patch from llvm-project PR 91524
git apply ../build_tools/llvm-project.patch

# Do the build
cd ..
./build_tools/scripts/build_mlir.sh llvm-project build/llvm-project
```

2. Build the project and run all tests

```bash
cmake -B ./build/mlir-tensorrt -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_TRT_USE_LINKER=lld \
    -DMLIR_TRT_PACKAGE_CACHE_DIR=${PWD}/.cache.cpm \
    -DMLIR_DIR=build/llvm-project/lib/cmake/mlir \
    -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON
ninja -C build/mlir-tensorrt all
ninja -C build/mlir-tensorrt check-mlir-executor
ninja -C build/mlir-tensorrt check-mlir-tensorrt-dialect
ninja -C build/mlir-tensorrt check-mlir-tensorrt
```

3. Build Python binding wheels

This will produce wheels under `build/mlir-tensorrt/wheels`:

```
ninja -C build/mlir-tensorrt mlir-tensorrt-all-wheels
```






