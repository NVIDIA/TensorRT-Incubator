# MLIR-TensorRT

**Goal: Provide inference acceleration for tensor programs which can be serialized as certain MLIR IRs (e.g. StableHLO) by offloading to TensorRT and other NVIDIA technologies. Generate kernels where needed to provide complete coverage of StableHLO, including support for bounded dynamism. The compiler is modular, enabling users with MLIR stacks to easily integrate individual components of the compiler.**

# Quickstart

You can download pre-compiled binaries and Python wheels on the [GitHub releases page](https://github.com/NVIDIA/TensorRT-Incubator/releases).

## Compiler

The compiler tool `mlir-tensorrt-compiler` compiles StableHLO MLIR programs.
The compiler will attempt to segment the StableHLO program and utilize different compilation backends such
as TensorRT or our own fallback kernel generator.  Backends are currently prioritized in
roughly that order. Each backend will produce an artifact which is either embedded directly in the
compiler MLIR output or emitted as a separate file. A host program in one of three different formats
is also emitted. See the table below for example commands:


| Host Output Option      | Description                              | Current Testing/Functionality Level | Example Command Line                                                          |
|----------------------|---------------------------------------------|-------------------------------------------------------------------------------|
| `mtrt-interpreter` (default) | Generate host code for the MTRT interpreter | best  |  `mlir-tensorrt-compiler input.mlir --opts="host-target=executor" -o=output.rtexe --entrypoint=main` |
| `cpp`/`emitc`        | Emit plain C++ host code                    | basic | `mlir-tensorrt-compiler input.mlir --opts="host-target=emitc" -o=output --entrypoint=main`           |
| `llvm`               | Emit LLVM-IR for the host, runnable with `mlir-cpu-runner` + MLIR-TRT C support library | basic | `mlir-tensorrt-compiler input.mlir --opts="host-target=llvm" -o=output.llvm --entrypoint=main`       |

Choose the output format which best matches your integration or development needs. How to use the output
will depend on the host output format. See the [Runtime](#runtime) section.

## Runtime

The runtime usage depends on the host output format selected during compilation:

### Executor Format (`host-target=executor`)

The compiler generates an executor runtime executable (`.rtexe` file) that can be executed using the `mlir-tensorrt-runner` tool:

```bash
mlir-tensorrt-runner output.rtexe --features=core,cuda,tensorrt
```

The executor runner supports various runtime features and can be configured with command-line options. See `mlir-tensorrt-runner --help` for details.

### C++ Format (`host-target=emitc`)

The compiler generates C++ source code along with TensorRT engine files and PTX modules. To use the generated code:

1. Compile the generated C++ file along with the MLIR-TensorRT runtime headers (found in `mtrt-runtime/`)
2. Link against TensorRT libraries (`libnvinfer.so`) and CUDA runtime
3. Ensure TensorRT engine files (`.trtengine`) and PTX modules (`.ptx`) are accessible at runtime

Example compilation:

```bash
g++ -I/path/to/mtrt-runtime output.cpp -lnvinfer -lcudart -o output_executable
```

The generated C++ code includes initialization functions (e.g., `*_initialize`) that must be called before inference, and cleanup functions (e.g., `*_destroy`) that should be called on shutdown.

### LLVM Format (`host-target=llvm`)

The compiler generates LLVM *MLIR* that can be JIT compiled and executed with standard MLIR
tools like `mlir-runner`:

```bash
mlir-runner output.mlir -e main --entry-point-result=i64 --shared-libs=libmtrt_runtime.so
```

# Components

MLIR-TensorRT is organized into several sub-projects. The `common` folder is common code that
is shared amongst all projects. The projects `executor`, `kernel`, and `tensorrt` are otherwise independent
and their purpose is described below:

1. `tensorrt`: contains an MLIR dialect that precisely models the TensorRT input language
   and provides optimization passes and translation from TensorRT MLIR to a TensorRT
   `libnvinfer` engine. Some users use only this component in their own MLIR-based compilers.

2. `executor`: contains an MLIR dialect that is a simplified form of LLVM-IR. It provides
   translations to MTRT Interpreter (Lua), LLVM-IR, and C++. Note that translation to Lua
   is an implementation detail, executing the Lua code requires our Lua runtime. Translated
   C++ requires a small C support library for compilation, as does LLVM IR output.

The remaining components utilize these previous three to build higher-level tools:

1. `compiler`: contains the top-level StableHLO compiler and consumes the other
   three projects as dependencies. It contains an MLIR pipeline that performs input
   preprocessing transformations, segmentation of the input program, dispatching segments
   to different compiler backends, and lowering down to the `executor` IR.
2. `integrations`: Contains Python bindings

# Development Instructions

See the [developer docs](./docs/Development.md).
