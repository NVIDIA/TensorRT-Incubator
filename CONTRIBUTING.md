# Contributing To TensorRT-Incubator Project

This repository currently host two projects i.e. [tripy](./README.md) and [mlir-tensorrt](../mlir-tensorrt/README.md).

For each project in the repository, create a separate PR.
For easier dependency management, ensure changes in `tripy` that depend on updates in `mlir-tensorrt` can only be merged after the corresponding changes in `mlir-tensorrt` are released and integrated.

Follow project specific contribution guidelines for more details:
  - [Contribute to Tripy](./tripy/CONTRIBUTING.md)
  - [Contribute to MLIR-TensorRT](./mlir-tensorrt/CONTRIBUTING.md)
