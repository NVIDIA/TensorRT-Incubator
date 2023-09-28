# Tripy (Tri-Py): TensorRT meets Python

##  What is Tripy

## Overview

## Installation

## Quick start

## Examples


## Developer instructions

### Setup instructions

```
docker build -t tripy_container .
docker run --gpus all  -it -v /path_to_tripy/tripy/:/tripy/ --rm tripy_container:latest

## Inside the container
poetry install
poetry run pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Hacks required:
1.  ```export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring```
2. `poetry update` requires enough space in `/tmp` directory. You can change the `/tmp` directory via `export TMPDIR=$HOME/tmp`.
