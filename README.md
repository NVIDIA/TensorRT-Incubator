# Tripy (Tri-Py): TensorRT meets Python

##  What is Tripy

## Overview

## Installation

## Quick start

## Examples


## Setup instructions

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

Hacks required:
1.  ```export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring```
2. `poetry update` requires enough space in `/tmp` directory. You can change the `/tmp` directory via `export TMPDIR=$HOME/tmp`.
