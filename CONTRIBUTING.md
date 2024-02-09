# Contributing To Tripy

## Setting Up

1. Clone the Tripy repository:

	```bash
	git clone ssh://git@gitlab-master.nvidia.com:12051/TensorRT/poc/tripy.git
	```

2. Pull the prebuilt development container:

	```bash
	docker login gitlab-master.nvidia.com:5005/tensorrt/poc/tripy
	docker pull gitlab-master.nvidia.com:5005/tensorrt/poc/tripy
	```

3. Launch the container; from the [`tripy` root directory](.), run:

	```bash
	docker run --gpus all -it -v $(pwd):/tripy/ --rm gitlab-master.nvidia.com:5005/tensorrt/poc/tripy:latest
	```

4. You should now be able to use `tripy` in the container. To test it out, you can run a quick sanity check:

	```bash
	python3 -c "import tripy as tp; print(tp.ones((2, 3)))"
	```

	This should give you some output like:
	```
	tensor(
		[[1. 1. 1.]
		[1. 1. 1.]],
		dtype=float32, loc=gpu:0, shape=(2, 3)
	)
	```

## Making Changes

### Before You Start: Install pre-commit

Before committing changes, make sure you install the pre-commit hook.
You only need to do this once.

We suggest you do this *outside* the container and also use `git` from
outside the container.

From the [`tripy` root directory](.), run:
```bash
python3 -m pip install pre-commit
pre-commit install
```

### Tests

Almost any change you make will require you to add tests or modify existing ones.
For details on tests, see [the tests README](./tests/README.md).


### Documentation

If you add or modify any public-facing interfaces, you should also update the documentation accordingly.
For details on the public documentation, see [the documentation README](./docs/README.md).


### Adding New Operators

If you want to add a new operator to Tripy, see the [guide](./docs/development/how-to-add-new-ops.md)
on how to do so.


## Advanced: Building A Container Locally

Generally, you should not need to build a container locally. If you do, you can follow these steps:

1. (Optional) Manually build `mlir-tensorrt` integration library.

	If you did not modify `mlir-tensorrt.txt`, you can skip this step.
	A script will automatically download the latest `mlir-tensorrt` package in Step 2.

	Building `mlir-tensorrt` is done in a separate container than `tripy` as eventually `mlir-tensorrt`
	will not be shipped externally and saves adding additional complexity to `tripy` containers.

	1. Get `mlir-tensorrt` repository:

		```bash
		git clone ssh://git@gitlab-master.nvidia.com:12051/TensorRT/poc/mlir/mlir-tensorrt.git
		cd mlir-tensorrt && git checkout $(cat ../mlir-tensorrt.txt)
		git submodule update --init --depth 1
		```

	2. Install docker-compose:

		```bash
		sudo apt-get install docker-compose
		```

	3. Build the `mlir-tensorrt` container locally:

		```bash
		cd build_tools/docker
		docker compose up -d
		```

	4. Copy your SSH key to the container. You can use `docker container ls` or `docker ps` to find the `<container-id>`

		Launch the container and create .ssh folder in /root.
		```bash
		docker compose exec mlir-tensorrt-poc-dev bash
		mkdir -p /root/.ssh
		```

		Now, copy SSH keys to the container.
		```bash
		docker cp ~/.ssh/id_rsa <container-id>:/root/.ssh
		```

	5. Launch the container:

		```bash
		docker compose exec mlir-tensorrt-poc-dev bash
		```

	6. Build `mlir-tensorrt`:

		```bash
		cd /workspaces/mlir-tensorrt/
		cmake -B build -S . -G Ninja \
			-DCMAKE_BUILD_TYPE=RelWithDebInfo \
			-DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
			-DCMAKE_C_COMPILER_LAUNCHER=ccache \
			-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
			-DLLVM_USE_LINKER=lld \
			-DMLIR_TRT_ENABLE_TRIPY=ON

		ninja -C build all
		```

	7. (Optional) To verify the build, the below command should dump out an .mlir file with tensorrt operations:

		```bash
		./build/tools/mlir-tensorrt-opt examples/matmul_mhlo.mlir \
			-pass-pipeline="builtin.module(func.func(convert-hlo-to-tensorrt{allow-i64-to-i32-conversion},tensorrt-expand-ops,translate-tensorrt-to-engine))" \
			-mlir-elide-elementsattrs-if-larger=128
		```

	After building `mlir-tensorrt` project, the build will be available in the `tripy` container.
	The integrated tripy library file is `libtripy_backend_lib.so`.

2. Download dependencies from CI.

	Download `stablehlo` and `mlir-tensorrt` (if you do not have to manully build it), the script will skip downloading if the packages already exist. If you want to download the latest build, make sure to remove the existing packages from the tripy directory.

	```bash
	export TRIPY_GITLAB_API_TOKEN=<your-access-token>
	python3 scripts/download_dependencies.py
	```

3. Build the tripy container.

	From the [`tripy` root directory](.), run:
	```bash
	docker build -t tripy .
	docker run --gpus all -it -v $(pwd):/tripy/ --rm tripy:latest
	```
