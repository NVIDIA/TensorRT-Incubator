# Contributing To Tripy

## Local Development

### Install pre-commit

From the [`tripy` root directory](.), run:
```bash
pip install pre-commit
pre-commit install
```

### Using A Prebuilt Container

From the [`tripy` root directory](.), run:
```bash
docker login gitlab-master.nvidia.com:5005/tensorrt/poc/tripy
docker pull gitlab-master.nvidia.com:5005/tensorrt/poc/tripy
docker run --gpus all -it -v $(pwd):/tripy/ --rm gitlab-master.nvidia.com:5005/tensorrt/poc/tripy:latest
```

### Building A Container Locally

From the [`tripy` root directory](.), run:
```bash
docker build -t tripy .
docker run --gpus all -it -v $(pwd):/tripy/ --rm tripy:latest
```

### Build mlir-tensorrt

Building `mlir-tensorrt` is done in a separate container than `tripy` as eventually `mlir-tensorrt` will not be shipped externally and saves adding additional complexity to `tripy` containers.

From the [`tripy` root directory](.), run:

Get `mlir-tensorrt` repository:
```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/TensorRT/poc/mlir/mlir-tensorrt.git
cd mlir-tensorrt && git checkout $(cat ../mlir-tensorrt.txt)
git submodule update --init --depth 1
```

Install docker-compose:
```bash
sudo apt-get install docker-compose
```

Launch `mlir-tensorrt` container and build `mlir-tensorrt`:
```bash
# Build mlir-tensorrt container locally
cd build_tools/docker
docker compose up -d
# copy ssh key to mlir-tensorrt container, use docker ps to find {container-id}
docker cp ~/.ssh/id_rsa {container-id}:/root/.ssh
# launch mlir-tensorrt container
docker compose exec mlir-tensorrt-poc-dev bash

cd /workspaces/mlir-tensorrt/
cmake -B build -S . -G Ninja \
	 -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
	 -DCMAKE_C_COMPILER_LAUNCHER=ccache \
	 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
	 -DLLVM_USE_LINKER=lld

ninja -C build all

# To verify the build, the below command should dump out .mlir file with tensorrt operations
./build/tools/mlir-tensorrt-opt examples/matmul_mhlo.mlir -pass-pipeline="builtin.module(func.func(convert-hlo-to-tensorrt{allow-i64-to-i32-conversion},tensorrt-expand-ops,translate-tensorrt-to-engine))" -mlir-elide-elementsattrs-if-larger=128
```

After building `mlir-tensorrt` project, the build will be available in the `tripy` container. The integrated tripy lib file is `libtripy_backend_lib.so`.

## Running Tests

You can run tests locally by running:
```bash
pytest -v
```

## Building Documentation

You can build the documentation locally by running:
```bash
sphinx-build docs build/docs -j 6 -W
```
