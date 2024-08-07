# Contributing To Tripy


## Setting Up

1. Clone the TensorRT-Incubator repository:

    ```bash
    git clone https://github.com/NVIDIA/TensorRT-Incubator.git
    ```

2.  From the [`tripy` root directory](.), run:

    <!-- TODO (#release) -->
    ```bash
    docker build -t tripy .
    ```

3. Launch the container; from the [`tripy` root directory](.), run:

    ```bash
    docker run --gpus all -it -v $(pwd):/tripy/ --rm tripy:latest
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

### Getting Up To Speed

We've added several guides [here](./docs/post0_developer_guides/) that can help you better understand
the codebase. We recommend starting with the [architecture](./docs/post0_developer_guides/architecture.md)
documentation.

If you're intersted in adding a new operator to Tripy, refer to [this guide](./docs/post0_developer_guides/how-to-add-new-ops.md).

### Making Commits

Ensure any commits you make are signed. See
[this page](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#ssh-commit-signature-verification)
for details on signing commits.

Please make sure any contributions you make satisfy the developer certificate of origin:

> Developer Certificate of Origin
>	Version 1.1
>
>	Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
>
>	Everyone is permitted to copy and distribute verbatim copies of this
>	license document, but changing it is not allowed.
>
>
>	Developer's Certificate of Origin 1.1
>
>	By making a contribution to this project, I certify that:
>
>	(a) The contribution was created in whole or in part by me and I
>		have the right to submit it under the open source license
>		indicated in the file; or
>
>	(b) The contribution is based upon previous work that, to the best
>		of my knowledge, is covered under an appropriate open source
>		license and I have the right under that license to submit that
>		work with modifications, whether created in whole or in part
>		by me, under the same open source license (unless I am
>		permitted to submit under a different license), as indicated
>		in the file; or
>
>	(c) The contribution was provided directly to me by some other
>		person who certified (a), (b) or (c) and I have not modified
>		it.
>
>	(d) I understand and agree that this project and the contribution
>		are public and that a record of the contribution (including all
>		personal information I submit with it, including my sign-off) is
>		maintained indefinitely and may be redistributed consistent with
>		this project or the open source license(s) involved.


### Use custom MLIR-TensorRT with Tripy
Tripy depends on [MLIR-TensorRT](../mlir-tensorrt/README.md) for compilation and execution.
The Tripy container currently builds with [mlir-tensorrt-v0.1.29](https://github.com/NVIDIA/TensorRT-Incubator/releases/tag/mlir-tensorrt-v0.1.29), but you may
choose to test Tripy with a custom version of MLIR-TensorRT.

1. [Build custom MLIR-TensorRT](#contributing-to-mlir-tensorrt)

2. Launch the container with mlir-tensorrt repository mapped for accessing wheels files; from the [`tripy` root directory](.), run:
    ```bash
    docker run --gpus all -it -v $(pwd):/tripy/ -v $(pwd)/../mlir-tensorrt:/mlir-tensorrt  --rm tripy:latest
    ```

3. Install MLIR-TensorRT wheels
    MLIR-TensorRT builds with a specific version of TensorRT, ensure it matches the version required by Tripy.
    This requirement can be relaxed in future.

    You can confirm the required TensorRT version:
    ```bash
      echo "$LD_LIBRARY_PATH" | grep -oP 'TensorRT-\K\d+\.\d+\.\d+\.\d+'
    ```
    Ensure that installed wheels version (`major * 10 + minor` version) matches above TensorRT version.

    For python 3.10.12, run:
    ```bash
    python3 -m pip install --force-reinstall /mlir-tensorrt/build/mlir-tensorrt/wheels/python3.10.12/trt101/**/*.whl
    ```

4. Verify everything works
    ```bash
    python3 -c "import tripy as tp; print(tp.ones((2, 3)))"
    ```

### Tests

Almost any change you make will require you to add tests or modify existing ones.
For details on tests, see [the tests README](./tests/README.md).

### Documentation

If you add or modify any public-facing interfaces, you should also update the documentation accordingly.
For details on the public documentation, see [the documentation README](./docs/README.md).
