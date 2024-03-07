# Contributing To Tripy


## Table Of Contents

[[_TOC_]]


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

### Getting Up To Speed

We've added several guides [here](./docs/development/) that can help you better understand
the codebase. We recommend starting with the [architecture](./docs/development/architecture.md)
documentation.

If you're intersted in adding a new operator to Tripy, refer to [this guide](./docs/development/how-to-add-new-ops.md).


### Tests

Almost any change you make will require you to add tests or modify existing ones.
For details on tests, see [the tests README](./tests/README.md).


### Documentation

If you add or modify any public-facing interfaces, you should also update the documentation accordingly.
For details on the public documentation, see [the documentation README](./docs/README.md).


## Advanced: Building A Container Locally

1. First [set up your personal access token in GitLab](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html)
	and export it:

	```bash
	export GITLAB_API_TOKEN=<your token here>
	```

2.  From the [`tripy` root directory](.), run:

	```bash
	docker build -t tripy --build-arg gitlab_user=__token__ --build-arg gitlab_token=$GITLAB_API_TOKEN  .
	```

3. You can launch the container with:

	```bash
	docker run --gpus all -it -v $(pwd):/tripy/ --rm tripy:latest
	```

## Advanced: Debugging

In order to use `lldb` in tripy container, launch the container with extra security options:

```bash
docker run --gpus all --cap-add=SYS_PTRACE \
	--security-opt seccomp=unconfined --security-opt apparmor=unconfined \
	-v $(pwd):/tripy/ -it --rm tripy:latest
```
See https://forums.swift.org/t/debugging-using-lldb/18046 for more details.
