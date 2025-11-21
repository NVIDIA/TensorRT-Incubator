### MLIR-TensorRT CI/CD Scripts

This directory contains convenience scripts used by CI/CD and for local development. They provide sensible defaults so you can run them with no arguments, and allow overriding behavior using environment variables.

### CI Images

- Release CI (build wheels and distribution tar.gz) uses Rocky-based images:
  - `ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-rocky-gcc11`
  - `ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-rocky`

- Pull Requests and Nightly CI use Ubuntu-based images:
  - `ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu-llvm17`
  - `ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda13.0-ubuntu`


### Run Docker Image

Ubuntu image (PR/nightly) example:


```bash
#Make sure you have permission to pull the image from ghcr.io
docker login ghcr.io

docker run -it --gpus all --user root --shm-size=10.24g \
  --ulimit stack=67108864 --ulimit memlock=-1 --cap-add SYS_ADMIN \
  --net host -v /var/run/docker.sock:/var/run/docker.sock \
  -v /path/to/mlir-tensorrt:/mlir-tensorrt \
  ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-ubuntu-llvm17 bash
```

Rocky image (release) example:

```bash
docker run -it --gpus all --user root --shm-size=10.24g \
  --ulimit stack=67108864 --ulimit memlock=-1 --cap-add SYS_ADMIN \
  --net host -v /var/run/docker.sock:/var/run/docker.sock \
  -v /path/to/mlir-tensorrt:/mlir-tensorrt \
  ghcr.io/nvidia/tensorrt-incubator/mlir-tensorrt:cuda12.9-rocky-gcc11 bash
```


### cicd_build_test.sh
Builds and optionally tests MLIR-TensorRT using CMake presets and Ninja.

- Default behavior:
  - Builds and runs tests.
  - Detects and uses a Python virtualenv via `uv` for dependencies.
- Key environment variables (overridable):
  - `REPO_ROOT`: repo root (auto-detected)
  - `BUILD_DIR`: build directory (default: `REPO_ROOT/build`)
  - `CMAKE_PRESET`: CMake preset (default: `github-cicd`)
  - `SKIP_TESTS`: skip tests (default: `0`; set to `1` to build only)
  - `DOWNLOAD_TENSORRT_VERSION`: preferred TensorRT version; if unset, falls back to `TENSORRT_VERSION`, else `10.12`
  - `CPM_SOURCE_CACHE`: (default: `REPO_ROOT/.cache.cpm`)
  - `CCACHE_DIR`: (default: `REPO_ROOT/ccache`)
  - `NINJA_JOBS`: parallel jobs for Ninja (default: `nproc`)
  - `VERBOSE`: verbose output (default: `0`)

Examples:

```bash
# Build and test (default)
./build_tools/scripts/cicd_build_test.sh

# Build only (skip tests)
SKIP_TESTS=1 ./build_tools/scripts/cicd_build_test.sh

# AddressSanitizer preset
CMAKE_PRESET=github-cicd-with-asan ./build_tools/scripts/cicd_build_test.sh
```


### cicd_build_distribution.sh
Builds a distribution tarball for MLIR-TensorRT. By default it reuses the CI build-and-test flow, then creates an archive under `PKG_DIR`.

- Default behavior:
  - `PKG_DIR`: `REPO_ROOT/install`
  - `PKG_FILE`: `mlir-tensorrt-$(uname -m)-linux-tensorrt${DOWNLOAD_TENSORRT_VERSION}.tar.gz`
  - `CMAKE_PRESET`: `distribution`
  - `SKIP_TESTS`: `1` (set to `0` to run tests before packaging)
  - `DOWNLOAD_TENSORRT_VERSION`: if unset, falls back to `TENSORRT_VERSION`, else `10.12`
  - `VERBOSE`: `0`
  - Compression uses `pigz` if available, else `gzip`.

Examples:

```bash
# Use all defaults and create the archive
./build_tools/scripts/cicd_build_distribution.sh

# Customize output name and version
PKG_FILE=mlir-tensorrt-x86_64-linux-tensorrt10.13.tar.gz \
DOWNLOAD_TENSORRT_VERSION=10.13 \
./build_tools/scripts/cicd_build_distribution.sh

# Run tests prior to packaging
SKIP_TESTS=0 ./build_tools/scripts/cicd_build_distribution.sh
```

Notes:
- The archive is written to `${PKG_DIR}/${PKG_FILE}` (unless `PKG_FILE` includes a path).
- Internals expect an install tree under `${PKG_DIR}/<folder>/mlir-tensorrt` (the script prepares this layout before archiving).


### cicd_build_wheels.sh
Builds Python wheels for MLIR-TensorRT integrations via `uv build`.

- Default behavior:
  - Builds wheels for all packages and Python versions.
  - Writes wheels to `WHEELS_DIR` (default: `REPO_ROOT/.wheels`).
- Key environment variables (overridable):
  - `WHEELS_DIR`: wheels output directory (default: `REPO_ROOT/.wheels`)
  - `PYTHON_VERSIONS` or `python_versions`: space/comma separated versions (default: `3.10 3.11 3.12 3.13`)
  - `PACKAGES`: space/comma separated list of packages (default: `mlir_tensorrt_tools mlir_tensorrt_compiler mlir_tensorrt_runtime`)
  - `DOWNLOAD_TENSORRT_VERSION`: if unset, falls back to `TENSORRT_VERSION`, else `10.12`
  - `MLIR_TRT_BUILD_DIR`: build directory (default: `REPO_ROOT/build`)
  - `VERBOSE`: verbose output (default: `0`)

Examples:

```bash
# Build all packages for all default Python versions into .wheels
./build_tools/scripts/cicd_build_wheels.sh

# Build only for Python 3.10 and 3.11
PYTHON_VERSIONS="3.10 3.11" ./build_tools/scripts/cicd_build_wheels.sh

# Build just the runtime package
PACKAGES=mlir_tensorrt_runtime ./build_tools/scripts/cicd_build_wheels.sh
```


### Prerequisites
- The Docker images used in CI already contain required toolchains. For local environments:
  - CMake, Ninja, Clang/LLVM, `uv`, Python toolchain(s)
  - Optional: `ccache`, `pigz` for faster compression


### Tips
- All scripts are safe to run from any working directory; they compute `REPO_ROOT` automatically.
- Most paths accept relative or absolute values; relative values are resolved against `REPO_ROOT`.


