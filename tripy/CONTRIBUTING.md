# Contributing To Tripy

Thanks for your interest in contributing to Tripy!

## Setting Up

1. Clone the repository:

    ```bash
    git clone https://github.com/NVIDIA/TensorRT-Incubator.git
    ```

2. Launch the development container.

    -  In most cases, you can pull an existing one:

        1. Log in to the registry. Use your GitHub username and a
            [personal access token](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry)
            as the password:

            ```bash
            docker login ghcr.io/nvidia/tensorrt-incubator
            ```

        2. Pull and launch the container. From the [`tripy` root directory](.), run:

            ```bash
            docker run --pull always --gpus all -it --cap-add=SYS_PTRACE -p 8080:8080 -v $(pwd):/tripy/ --rm ghcr.io/nvidia/tensorrt-incubator/tripy
            ```

    - If you made changes to the container
        (e.g. changing [Dockerfile](./Dockerfile) or [pyproject.toml](./pyproject.toml)),
        build it locally and launch it. From the [`tripy` root directory](.), run:

        ```bash
        docker build -t tripy .
        docker run --gpus all -it --cap-add=SYS_PTRACE -p 8080:8080 -v $(pwd):/tripy/ --rm tripy:latest
        ```

    - If you are using Visual Studio Code, you can alternatively use the included `.devcontainer` configuration.

3. **[Optional]** Run a sanity check in the container:

    ```bash
    python3 -c "import nvtripy as tp; print(tp.ones((2, 3)))"
    ```

    You should see output like:
    ```
    tensor(
        [[1. 1. 1.]
        [1. 1. 1.]],
        dtype=float32, loc=gpu:0, shape=(2, 3)
    )
    ```

## Making Changes

### Before You Start: Install pre-commit

> [!TIP]
> Install the hook and use `git` *outside* the container
> so you don't have to repeat this step each time you launch the container.

From the [`tripy` root directory](.), run:
```bash
python3 -m pip install pre-commit
pre-commit install
```

### Getting Up To Speed

We've written developer guides to help you understand the codebase:

- [Architecture guide](https://nvidia.github.io/TensorRT-Incubator/post0_developer_guides/00-architecture.html)

- [Adding New Operations](https://nvidia.github.io/TensorRT-Incubator/post0_developer_guides/01-how-to-add-new-ops.html)


### Tests

See [the tests README](./tests/README.md) for details on adding tests for your changes.

### Documentation

If you add/modify any public-facing interfaces, update the documentation - see [this README](./docs/README.md).


### Making Commits

1. Set up
    [commit signing](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#ssh-commit-signature-verification).

2. Tripy follows [fork based developement](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).
    - Create a fork on GitHub. Make `git` aware of it with:

        ```bash
        git remote add fork <FORK_URL>
        ```

    - Create a feature branch. For example:

        ```bash
        git checkout -b my-feature-branch
        ```

    - After committing changes, push to your fork:

        ```bash
        git push fork
        ```

    - Create a
        [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
        on GitHub with a brief description explaining the change.

3. Contributions you make should satisfy the developer certificate of origin:

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


### Advanced: Using Custom MLIR-TensorRT Builds With Tripy

Tripy depends on [MLIR-TensorRT](../mlir-tensorrt/README.md) for compilation and execution.
The Tripy container includes a build of MLIR-TensorRT, but in some cases, you may want to test Tripy with a local build:

1. Build MLIR-TensorRT as per the instructions in the [README](../mlir-tensorrt/README.md).

2. Launch the container with mlir-tensorrt repository mapped for accessing wheels files; from the [`tripy` root directory](.), run:
    ```bash
    docker run --gpus all -it --cap-add=SYS_PTRACE -p 8080:8080 -v $(pwd):/tripy/ -v $(pwd)/../mlir-tensorrt:/mlir-tensorrt  --rm tripy:latest
    ```

3. Install MLIR-TensorRT wheels
    MLIR-TensorRT must be built against a specific version of TensorRT, but will be able
    to work with any ABI-compatible version at runtime.

    The Tripy container includes TensorRT. Follow these steps to confirm
    the TensorRT version and ensure compatibility with your TensorRT wheels:

    ```bash
    echo "$LD_LIBRARY_PATH" | grep -oP 'TensorRT-\K\d+\.\d+\.\d+\.\d+'
    ```

    Ensure the installed MLIR-TensorRT wheels have:
    * The same major version as the TensorRT version in the container.
    * A minor version equal to or higher than the version in the container.

    For example, to install Python 3.10.12 wheels compatible with TensorRT 10.2+, run:
    ```bash
    python3 -m pip install --force-reinstall /mlir-tensorrt/build/wheels/python3.10.12/trt102/**/*.whl
    ```

4. Verify everything works:
    ```bash
    python3 -c "import nvtripy as tp; print(tp.ones((2, 3)))"
    ```
