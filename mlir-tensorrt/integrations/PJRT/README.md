# MLIR-TensorRT JAX Plugin

The `mlir_tensorrt_jax` package provides an XLA PJRT plugin, which is
a compiler/runtime implementation which is used in place of or in
conjunction with XLA's GPU or CPU backends.

The PJRT Plugin uses MLIR-TensorRT to divide the program into different
backends such as TensorRT or direct kernel generation when TensorRT offload
is not possible.

This version was tested against JAX 0.5.1 but may be compatible with other JAX
`0.5.x` versions within the PJRT compatibility window.

# Installation

Installation is performed by using `pip` or `uv pip` commands to install
one of the pre-compiled wheels that matches your desired platform.

Two sets of wheels are provided corresponding to two different
platforms:

- x86 Linux with CUDA 12.9 and TensorRT 10.12 GA. These wheels were compiled on
  Rocky Linux 8 and therefore should be compatible with Linux systems
  using GLibC>=2.28.
- aarch64 NVIDIA Jetson Jetpack Linux 7, CUDA 13.0 and TensorRT 10.13 EA.

# Usage

Note that TensorRT (`libnvinfer.so`) and the CUDA runtime (`libcudart.so`) are dynamically
linked, so paths to these libraries must be added to the environment's
dynamic load path (e.g. via `LD_LIBRARY_PATH`).

JAX can load multiple PJRT plugin implementations at runtime, so
to ensure that the MLIR-TensorRT plugin is used, the user should
direct JAX to use the `mlir_tensorrt` plugin. This can be done using
any of two different methods:

1. Run `jax.config.update("jax_platforms", "my_plugin")` at the beginning
   of your program.
2. Set the environment variable `JAX_PLATFORMS=mlir_tensorrt` prior to running
   your program.

# Configuration

There are some options that can be set to control the
compiler behavior. These should be passed via the environment variable
`MLIR_TRT_FLAGS`.

1. `--tensorrt-builder-opt-level=[value]` -- This affects the optimization effort taken
  by the TensorRT backend. takes a numeric value from 0 (fastest
  compilation but worst performance) to 5 (slowest compilation, fastest performance).
  Default value is `0` since that provides the best user experience during model
  development and testing. For final inference/deployment uses, set this to `3` or above.

2. `--mtrt-pjrt-opt-level=[value]`  -- This affects optimizations applied by the MLIR-TensorRT
   Stablehlo compiler. Examples include loop unrolling. It takes a numeric value from 0
   to 5, just like `tensorrt-builder-opt-level`. The default value is `0`. At values of `1` and above,
   loops with statically known trip-counts are *aggressively unrolled*, which can dramatically increase
   compilation time but result in large performance gains.

3. `--mlir-pass-pipeline-crash-reproducer=crash.mlir` -- In the case of a compilation failure
   or a crash, you can generate a reproducer file to send with your bug report. This option will
   generate a `crash.mlir` file in the current working directory if the MLIR-TRT compilation
   pipeline fails.

Example of setting flags:

```
export MLIR_TRT_FLAGS="--tensorrt-builder-opt-level=3 --mtrt-pjrt-opt-level=3 --mlir-pass-pipeline-crash-reproducer=crash.mlir"
```

# Limitations

The MLIR-TensorRT JAX Plugin currently has the following limitations:

1. A few JAX operations are supported. This includes operations like
   FFT and `lax.linalg.triangular_solve`.
2. Triton/Pallas
3. Multi-GPU/NCCL operations
4. JAX donation arguments

