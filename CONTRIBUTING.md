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

## Running Tests

You can run tests locally by running:
```bash
poetry run pytest -v
```

## Building Documentation

You can build the documentation locally by running:
```bash
poetry run sphinx-build docs build/docs -j 6 -W
```
