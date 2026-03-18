# MLIR-TensorRT

MLIR-TensorRT is a compiler and runtime for accelerating tensor programs on
NVIDIA GPUs. It takes programs expressed in
[StableHLO](https://github.com/openxla/stablehlo) (or other MLIR dialects),
partitions them across compilation backends — primarily
[TensorRT](https://developer.nvidia.com/tensorrt) and a fallback GPU kernel
generator — and emits a self-contained executable.

The project is built on the [MLIR](https://mlir.llvm.org/) compiler
infrastructure and is organized as a set of composable sub-projects so that
individual components (for example, the TensorRT dialect alone) can be
integrated into other MLIR-based compilers.

## Quickstart

Pre-compiled binaries and Python wheels are published on the
[GitHub releases page](https://github.com/NVIDIA/TensorRT-Incubator/releases).

### Compiling a StableHLO program

The main entry point is the `mlir-tensorrt-compiler` tool. It accepts a
StableHLO MLIR file and produces output in one of three host-code formats:

| Host Target | Description | Example |
|---|---|---|
| `executor` (default) | Flatbuffer interpreted by the MTRT executor runtime | `mlir-tensorrt-compiler input.mlir --opts="host-target=executor" -o output.rtexe --entrypoint=main` |
| `emitc` | Plain C++ source (compiled separately with a C++ compiler) | `mlir-tensorrt-compiler input.mlir --opts="host-target=emitc" -o output --entrypoint=main` |
| `llvm` | LLVM MLIR for JIT compilation (experimental) | `mlir-tensorrt-compiler input.mlir --opts="host-target=llvm" -o output.llvm --entrypoint=main` |

### Running the compiled output

**Executor format** — run with `mlir-tensorrt-runner`:

```bash
mlir-tensorrt-runner output.rtexe --features=core,cuda,tensorrt
```

**C++ format** — compile the generated source and link against the runtime:

```bash
g++ -I/path/to/mtrt-runtime output.cpp -lnvinfer -lcudart -o output_executable
```

**LLVM format** — JIT-execute with `mlir-runner`:

```bash
mlir-runner output.mlir -e main --entry-point-result=i64 --shared-libs=libmtrt_runtime.so
```

## Architecture

The compiler pipeline processes a StableHLO program through five phases:

1. **Setup** — canonicalize input, generate ABI wrappers, assign backend metadata.
2. **Input** — lower CHLO to StableHLO, refine shapes, fold constants.
3. **Clustering** — segment the program into clusters and assign each cluster to
   a backend (TensorRT, GPU kernel generator, or host).
4. **Bufferization** — convert tensors to explicit memory allocations.
5. **Lowering** — translate to the chosen host-code format (Executor, EmitC, or
   LLVM IR).

Backends are selected by benefit: TensorRT is preferred when it can handle an
operation; the kernel generator covers the remainder; simple scalar/shape
operations run on the host.

See [docs/StableHLOCompiler.md](./docs/StableHLOCompiler.md) for a detailed
description of the compilation pipeline.

## Project Structure

The `common/` directory contains shared infrastructure (custom TableGen
backends, dialect interfaces, the TensorRT dynamic loader). The remaining
sub-projects are:

| Sub-project | Description |
|---|---|
| [`tensorrt/`](./tensorrt/README.md) | MLIR dialect that precisely models the TensorRT API. Provides validation, optimization passes, and translation from MLIR to serialized TensorRT engines. Can be used independently in other MLIR compilers. |
| `executor/` | MLIR dialect representing a simplified form of LLVM-IR. Provides translation to Lua (for the MTRT interpreter runtime), C++, and LLVM IR. Also contains the Lua-based executor runtime and its C support library. |
| `kernel/` | GPU kernel generation backend. Lowers supported operations via Linalg and Transform IR to NVVM and then to PTX. |
| [`compiler/`](./docs/StableHLOCompiler.md) | Top-level StableHLO compiler. Orchestrates the pipeline described above, consuming the other sub-projects as dependencies. Contains the Plan dialect for clustering and segmentation. |
| `integrations/` | Python bindings (`mlir_tensorrt_compiler`, `mlir_tensorrt_runtime`) and a PJRT plugin for JAX interoperability. |

## Key Tools

| Tool | Purpose |
|---|---|
| `mlir-tensorrt-compiler` | End-to-end StableHLO compiler |
| `mlir-tensorrt-runner` | Execute `.rtexe` files produced by the compiler |
| `mlir-tensorrt-opt` | Run individual MLIR passes |
| `mlir-tensorrt-translate` | Run MLIR translations (e.g. to TensorRT engines) |
| `tensorrt-opt` | Run TensorRT-dialect-specific passes |

## Dependencies

- **LLVM / MLIR** — compiler infrastructure
- **StableHLO** — input dialect
- **TensorRT** (`libnvinfer`) — inference engine (version 10.x)
- **CUDA Toolkit** — GPU runtime, PTX compilation
- **Flatbuffers** — serialization for executor executables
- **Lua 5.4 / Sol2** — executor runtime interpreter

Optional: cuBLAS, NCCL (multi-GPU), torch-mlir.

## Development

See the [developer documentation](./docs/Developers.md) for build
instructions, testing, and contribution guidelines.
