# StableHLO Compiler

The StableHLO compiler is the top-level, end-to-end compilation pipeline in
MLIR-TensorRT. It takes a [StableHLO](https://github.com/openxla/stablehlo)
program, partitions it across compilation backends, and emits a complete
executable in one of three host-code formats.

## Overview

A typical compilation looks like this:

```
StableHLO MLIR
    │
    ▼
┌────────────────────┐
│  Setup & Input     │  canonicalize, shape refinement, constant folding
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Clustering        │  segment ops → TensorRT / Kernel / Host clusters
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Bufferization     │  tensors → explicit memory allocations
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Lowering          │  emit host code + embedded artifacts
└────────┬───────────┘
         │
    ┌────┴────┬──────────┐
    ▼         ▼          ▼
 .rtexe     C++ src    LLVM MLIR
```

The compiler is implemented in `compiler/lib/Compiler/Pipeline.cpp` and uses an
extension mechanism so that backends (TensorRT, kernel generator) can hook into
the pipeline at well-defined extension points.

## Pipeline Phases

### Phase 1: Setup

Prepares the module for compilation:

- Convert VHLO to StableHLO (if the input uses the versioned format).
- Assign default backend metadata (`plan.populate-default-backend-metadata`).
- Legalize I/O bounds attributes.
- Verify inputs and assign buffer slots.
- Generate ABI wrappers for the executor runtime (when using the executor host
  target).

### Phase 2: Input

Transforms the StableHLO input into a canonical form suitable for clustering:

- Lower CHLO to StableHLO, then lower special custom calls and composites.
- Inline functions and expand tuples.
- Refine shapes, canonicalize dynamism, and fold constants (with extension hooks
  so that backends can register custom constant-folding rules).
- Apply target-specific optimizations.
- Convert StableHLO control flow to SCF where appropriate.

### Phase 3: Clustering

This is the core partitioning phase, driven by the **Plan dialect**:

1. **Materialize shape calculations** — hoist shape computations so they are
   available to guide clustering.
2. **Create shape functions** — factor out shape computation into separate
   functions.
3. **Cluster** — group operations into clusters based on which backend can
   handle them. Each backend exposes a *clusterability* predicate and a
   *benefit* score. Operations are assigned to the backend with the highest
   benefit that supports them.
4. **Outline clusters** — extract each cluster into its own function.
5. **Post-clustering validation** — verify that the partitioning is legal.

### Phase 4: Bufferization

Converts the tensor-level IR into an explicit memory representation:

- Assign memory spaces (host vs. device).
- Insert explicit host↔device transfers.
- Allocate tensors (`plan.alloc-tensors`) and run module-level bufferization.
- Deallocate buffers and hoist allocations where possible.

### Phase 5: Lowering

Translates the bufferized IR into the chosen host-code format and embeds or
emits backend artifacts (TensorRT engines, PTX modules):

- Lower Linalg/memref operations to CUDA.
- Optionally schedule asynchronous execution.
- Emit host code via one of three paths (see [Output Formats](#output-formats)).
- Serialize artifacts (engines, PTX) into the output.

## Backends

Backends implement the `CompilerBackendAttrInterface` and are selected during
the clustering phase based on benefit. The default priority order is:

### TensorRT (benefit 3)

Operations that can be represented in the TensorRT dialect are clustered
together and compiled into serialized TensorRT engines. This backend:

- Converts StableHLO operations to `tensorrt.*` operations.
- Runs the TensorRT dialect's transformation pipeline (see
  [tensorrt/README.md](../tensorrt/README.md)).
- Translates the resulting IR to a serialized TensorRT engine.
- Wraps the engine in a `TensorRTRuntime` call for the host program.

See `compiler/lib/Backends/TensorRT/` for the implementation.

### Kernel Generator (benefit 2)

Operations that cannot go to TensorRT but implement `TilingInterface` or can be
lowered through Linalg are compiled into GPU kernels. This backend:

- Lowers clusters through Linalg and applies Transform-IR-based tiling and
  fusion.
- Translates to NVVM IR and then to PTX via the `kernel` sub-project.
- The PTX module is embedded in the output as an artifact.

See `compiler/lib/Backends/Kernel/` for the implementation.

### Host (benefit 1)

Simple operations (scalar arithmetic, shape computations, reshapes) that do not
benefit from GPU execution are lowered directly to host code via standard
MLIR conversions (StableHLO → Linalg → loops → SCF).

See `compiler/lib/Backends/Host/` for the implementation.

## Output Formats

The compiler supports three host-code output formats, selected via the
`host-target` option:

### Executor Flatbuffer (`host-target=executor`)

This is the default and most mature format. The compiler:

1. Lowers to **Executor dialect** IR, a simplified form of LLVM-IR designed for
   interpretation.
2. Translates Executor IR to **Lua source code**.
3. Packages the Lua source, serialized TensorRT engines, PTX modules, constant
   data, and function metadata into a **Flatbuffer** (`.rtexe` file).

The `.rtexe` file is executed by `mlir-tensorrt-runner`, which loads the
Flatbuffer and runs the Lua code via an embedded Lua 5.4 interpreter. The
interpreter has access to runtime modules for CUDA, TensorRT, cuBLAS, and
optionally NCCL.

```bash
mlir-tensorrt-compiler input.mlir --opts="host-target=executor" -o output.rtexe
mlir-tensorrt-runner output.rtexe --features=core,cuda,tensorrt
```

### C++ / EmitC (`host-target=emitc`)

The compiler converts host code to EmitC operations and then translates to
plain C++ source. TensorRT engines and PTX modules are emitted as separate
files alongside the C++ source.

The generated code can be compiled with any C++ compiler and linked against
TensorRT, CUDA, and the MLIR-TensorRT C support library.

```bash
mlir-tensorrt-compiler input.mlir --opts="host-target=emitc" -o output
g++ -I/path/to/mtrt-runtime output.cpp -lnvinfer -lcudart -o executable
```

### LLVM IR (`host-target=llvm`)

*Experimental.* The compiler lowers to LLVM dialect MLIR, which can be
JIT-compiled and executed with `mlir-runner` and the MLIR-TensorRT shared
runtime library.

```bash
mlir-tensorrt-compiler input.mlir --opts="host-target=llvm" -o output.llvm
mlir-runner output.mlir -e main --entry-point-result=i64 --shared-libs=libmtrt_runtime.so
```

## Extension Points

The compiler pipeline exposes extension points that allow backends to inject
custom passes at well-defined stages:

| Extension Point | Phase | Purpose |
|---|---|---|
| `ConstantFolding` | Input | Register custom constant-folding rules |
| `PreClustering` | Clustering | Run passes before clustering decisions |
| `PostClustering` | Clustering | Lower cluster contents to backend-specific IR |
| `PreBufferization` | Bufferization | Prepare for bufferization (e.g. build TensorRT engines) |
| `PostBufferization` | Bufferization | Post-bufferization transformations |
| `ExecutorLowering` | Lowering | Inject passes into executor lowering |

Two built-in extensions use these hooks:

- **TensorRTExtension** — converts StableHLO clusters to TensorRT IR, builds
  engines, and lowers to TensorRTRuntime calls.
- **KernelGenExtension** — lowers clusters through Linalg and Transform IR to
  PTX.

## The Plan Dialect

The **Plan dialect** (`compiler/lib/Dialect/Plan/`) is the compiler's internal
dialect for managing clustering and segmentation. Key operations:

- **`plan.cluster`** — marks a cluster of operations assigned to a specific
  backend.
- **`plan.dps_cluster`** — a cluster with explicit capture and
  destination-passing-style semantics.
- **`plan.alloc_cluster`** — an isolated cluster where the callee allocates its
  own results.

Key passes:

- `plan-clustering` — the main clustering algorithm.
- `plan-create-closed-regions` — close cluster regions by adding explicit
  captures.
- `plan-outline-clusters` — outline each cluster into its own function.
- `plan-materialize-shape-calculations` — hoist shape computations.
- `plan-alloc-tensors` / `plan-module-bufferize` — tensor allocation and
  bufferization.

## Compiler Options

The compiler exposes options through the `--opts` flag or via the C++/Python
API:

Pipeline options are passed via `--opts="..."` on the command line:

| Option | Description |
|---|---|
| `host-target` | Output format: `executor` (default), `emitc`, `llvm` |
| `phase-start` / `phase-end` | Run only a subset of pipeline phases |
| `backends` | Override backend selection and priority |

Top-level flags:

| Flag | Description |
|---|---|
| `--entrypoint` | Entry-point function name |
| `-o` | Output file path |

## Python API

The `mlir_tensorrt_compiler` Python package provides programmatic access:

```python
import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir

with ir.Context() as context:
    m = ir.Module.parse(stablehlo_asm)

    client = compiler.CompilerClient(context)
    pipeline = client.get_pipeline(
        ["--tensorrt-builder-opt-level=0"],
    )
    pipeline.run(m.operation)
    exe = compiler.translate_mlir_to_executable(m.operation)
```

## Directory Layout

The tree below shows the key directories (some ancillary directories are
omitted for brevity):

```
compiler/
├── include/mlir-tensorrt/
│   └── Compiler/
│       ├── Options.h              # Compiler option definitions
│       ├── Pipeline.h             # Pipeline class and ExtensionPoint enum
│       ├── Extension.h            # Extension base class
│       ├── Client.h               # CompilerClient API
│       ├── Backends/              # Backend interface declarations
│       ├── Dialect/
│       │   ├── Plan/              # Plan dialect (clustering/segmentation)
│       │   ├── CUDA/              # CUDA dialect extensions
│       │   └── TensorRTRuntime/   # TensorRT runtime call dialect
│       └── Passes/
│           ├── 1_Core/            # Core pass declarations
│           └── 2_Host/            # Host lowering pass declarations
├── lib/
│   ├── Compiler/
│   │   ├── Pipeline.cpp           # Main pipeline implementation
│   │   ├── InputPipelines/        # StableHLO, Linalg input pipelines
│   │   └── Extensions/           # TensorRTExtension, KernelGenExtension
│   ├── Backends/
│   │   ├── TensorRT/              # TensorRT backend
│   │   ├── Kernel/                # Kernel generation backend
│   │   └── Host/                  # Host backend
│   ├── Dialect/                   # Plan, CUDA, TensorRTRuntime implementations
│   └── Passes/
│       ├── 0_Frontend/            # Input-level passes
│       ├── 1_Core/                # Core transformation passes
│       └── 2_Host/                # Host lowering (EmitC, LLVM)
├── test/                          # LIT/FileCheck tests
└── tools/
    ├── mlir-tensorrt-compiler/    # Main compiler binary
    ├── mlir-tensorrt-runner/       # Executor runner binary
    ├── mlir-tensorrt-opt/         # Pass runner
    └── mlir-tensorrt-translate/   # Translation tool
```
