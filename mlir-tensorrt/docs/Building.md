# Building MLIR-TensorRT

MLIR-TensorRT currently only supports using CMake.

## Get the code

There are two options for obtaining the code:

<details>
<summary><strong>Option 1: Clone the Full Repository (default)</strong></summary>

This method checks out the entire repository, including all projects and documentation.

```bash
git clone https://github.com/NVIDIA/TensorRT-Incubator.git
cd TensorRT-Incubator/mlir-tensorrt
```

</details>

<details>
<summary><strong>Option 2: Sparse Checkout (only MLIR-TensorRT and CI)</strong></summary>

If you wish to avoid downloading projects unrelated to MLIR-TensorRT (such as TriPy), you can use git's sparse-checkout feature to only pull the <code>mlir-tensorrt</code> folder and GitHub CI files. This reduces both clone time and disk usage:

```bash
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/TensorRT-Incubator.git
cd TensorRT-Incubator
git sparse-checkout init --cone
git sparse-checkout set mlir-tensorrt .github
git checkout main
cd mlir-tensorrt
```

This will ensure only the relevant files for MLIR-TensorRT (and the <code>.github</code> workflows) are checked out in your working directory.

</details>


## Setup your environment

### Option A (Recommended): Use the pre-configured development container and VSCode/Cursor

To simplify setting up the developer environment and managing dependencies, we
provide a [development container](./DevContainer.md).
Follow the instructions there, then go to the section on
installing [Python dependencies](#installing-python-dependencies).

### Option B: Setup manually

Power users or people who don't want to develop in a container can certainly also
set things up manually. Here's what you need. In the rest of the doc, we assume
you have the recommended items.

- CMake (minimum version specified as the first line in the [CMakeLists.txt](../CMakeLists.txt)).
- [`uv` for managing certain dependencies (e.g. Python)](https://docs.astral.sh/uv).
- **Recommended:** Use a recent version of LLVM-based C++ toolchain (e.g. `clang`, `lld`, `lldb`, etc). In the
  devcontainer we use the latest stable LLVM toolchain.
- **Recommended:** Use `ccache` to speed up repeated builds (`apt-get install ccache`).
- **Recommended:** Use `ninja` build generator for CMake. You can download it from their [GitHub releases page](https://github.com/ninja-build/ninja/releases).

### Installing Python dependencies.

Even if using the development container, you will need to install
Python dependencies prior to building:

```shell
uv sync --extra cu12 # or 'cu13' depending on CUDA version installed
source .venv/bin/activate
```

## Configuring the build

```bash
cmake --preset ninja-llvm
```

See the [CMake options](../CMakeOptions.cmake) or [CMake Presets](../CMakePresets.json) to learn
about all the things you can turn on and off when running the `cmake` command.

### Create a `.env` file

We recommend creating a `.env` file that contains:

```bash
#!/usr/bin/env bash
# Activate the Python venv setup by 'uv sync'.
SCRIPT_DIR=/workspaces/TensorRT-Incubator/mlir-tensorrt
source ${SCRIPT_DIR}/.venv/bin/activate

build_dir="$SCRIPT_DIR/build"
llvm_bin_path=$(find ${build_dir} -type f -name llvm-lit -exec dirname {} \;)
mlir_trt_bin_path=$build_dir/bin
export PATH=${llvm_bin_path}:${mlir_trt_bin_path}:$PATH

# It's useful to put the `libnvinfer.so` downloaded by CMake
# on your LD_LIBRARY_PATH:
DEFAULT_TRT_VERSION=10.13.0.35
trt_nvinfer_so_path=$SCRIPT_DIR/.cache.cpm/tensorrt/${DEFAULT_TRT_VERSION}-x86_64-Linux/lib
export LD_LIBRARY_PATH=${trt_nvinfer_so_path}:${LD_LIBRARY_PATH}
```

## Building

```bash
ninja -C build all
```

## Running tests

To run all tests for all mlir-tensorrt subprojects:

```bash
ninja -C build check-all-mlir-tensorrt
```

For more refined test, selection, you can use any of the below shorthand commands:

```bash
# Any of these commands run a subset of test for the corresponding sub-project:
ninja -C build check-mlir-executor # 'executor/test/'
ninja -C build check-mlir-tensorrt-dialect # 'tensorrt/test'
ninja -C build check-mlir-tensorrt # 'compiler/test/'
```

For very specific selection of individual tests, you can
set environment variable `LIT_FILTER` to a regex pattern
like in the following example:

```bash
LIT_FILTER=".*IntegrationTests.*" ninja -C build check-mlir-tensorrt
```

Or you can use the `llvm-lit` tool directly.

```bash
# Put the directory on your path containing llvm-lit. See above
# example .env file.

# Run all mlir-executor tests.
llvm-lit -vv build/executor/test

# Example selection of all mlir-executor integration tests:
llvm-lit -vv build/executor/test --filter='.*Integration.*'
```

## Further Information

### Create a `CMakeUserPresets.json`

The most common CMake configurations for the project are exposed
in `CMakePresets.json`. Sometimes, you want to add additional flags
to enable an optional feature (e.g. `-DMLIR_TRT_ENABLE_TORCH=ON`).
You can specify those directly on the `cmake` command line, but if you
have several flags it can be easier to create a custom preset.

Since `CMakePresets.json` is checked into Git, it's recommended
to create a `CMakeUserPresets.json` file, which allows you
to configure new CMake presets based off of the existing ones in
`CMakePresets.json`. The `CMakeUserPresets.json` file is not
checked into Git.

Here's an example of what should go in `CMakeUserPresets.json`:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "my-config",
      "displayName": "User build config",
      "inherits": "ninja-llvm",
      "cacheVariables": {
        "MY_FEATURE_VAR": "ON",
		    "MY_OTHER_OPTION": "some-value"
      }
    }
  ]
}
```

And you can then configure the project using:

```
cmake --preset my-config
```

### Using the `mold` linker

The default build configurations use the LLVM toolchain's `lld` as the linker,
which is significantly faster than building the project using `ld`. For
additional speedup, you can also try using `mold` as the linker.
Mold claims to be faster than other linkers: https://github.com/rui314/mold.

**Mold linker is already installed in all devcontainers**.

In this project, link times can be a significant portion of the overall
compilation time, and `mold` can provide significant speedup based on our
experience. However, for building release binaries, we use `lld`.

To use mold as the linker, you just point CMake
to the new toolchain file. This is most easily done in a `CMakeUserPresets.json`:


```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "my-config",
      "displayName": "User build config",
      "inherits": "ninja-llvm",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "${sourceDir}/build_tools/cmake/toolchains/host-llvm-mold.cmake"
      }
    }
  ]
}
```

Note that switching toolchain files requires a clean/fresh CMake configuration:

```
cmake --preset my-config --fresh
```


### Ensure clangd LSP sever integration is working properly

In VSCode or other editor with LSP integration, make sure `clangd` server is running. Autocompletion and `Go To Definition` should work (after opening a `.cpp` or `.h` file and clangd performs indexing). You half to run the build configuration command first before auto complete will work.

### Building with AddressSanitizer enabled

Building the project with `ASan|TSan|UBSan` enabled can be an effective way to
root out issues or diagnose bugs such as segfaults without
using a debugger such as `lldb`. To build the project with `AddressSanitizer`
(ASAN) enabled, add the config flag `-DENABLE_ASAN=ON` to your CMake
configuration command.

See the [`Sanitizers.cmake`](../build_tools/cmake/Sanitizers.cmake) file for
the available options. These only have an effect if MLIR-TensorRT is the top-level
project (and not being built as a submodule of another CMake project).

### Controlling the TensorRT version used for building and testing

There are two (mutually exclusive) ways of controlling the TensorRT version that the
project is built and tested with:

1. The primary and most convenient method is setting the
   `MLIR_TRT_DOWNLOAD_TENSORRT_VERSION` CMake variable. This directs the build system to
    download a particular TensorRT version on the fly.
	You can change TensorRT versions by simply reconfiguring with this variable
	to a different version. To see what versions are supported, look at the
    [cmake script](../build_tools/cmake/TensorRTDownloadURL.cmake).
2. One can configure the project to use any TensorRT version by setting the `MLIR_TRT_TENSORRT_DIR`
   variable to a particular TensorRT installation directory.

### Building Python packages

Building Python packages is best done using the `uv` tool.

```bash
DOWNLOAD_TENSORRT_VERSION=10.13 uv build integrations/python/mlir_tensorrt_compiler --wheel --out-dir dist
DOWNLOAD_TENSORRT_VERSION=10.13 uv build integrations/python/mlir_tensorrt_runtime --wheel --out-dir dist
```
