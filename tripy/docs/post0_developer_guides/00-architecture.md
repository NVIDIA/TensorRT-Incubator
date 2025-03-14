# Architecture

## Overview

Tripy builds an **MLIR** program by **tracing** functional-style **Python APIs**.

The program is compiled and executed by
[MLIR-TRT](https://github.com/NVIDIA/TensorRT-Incubator/tree/main/mlir-tensorrt):


```mermaid
graph TD
    subgraph "Tripy (Python)"
        subgraph "Frontend"
            A["Tripy Python API"]:::frontend
        end

        subgraph "Trace"
            A -->|Stage Out| B["Trace"]:::trace
        end

        subgraph "Backend"
            B --> C["MLIR (tensorrt dialect)"]:::backend
        end
    end

    subgraph "MLIR-TRT (C++)"
        C --> D["MLIR-TRT Compiler/Runtime"]:::mlirtrt
    end

    classDef frontend fill:#1E90FF,stroke:#000,stroke-width:2px;
    classDef trace fill:#9370DB,stroke:#000,stroke-width:2px;
    classDef backend fill:#32CD32,stroke:#000,stroke-width:2px;
    classDef mlirtrt fill:#FF6347,stroke:#000,stroke-width:2px;
```

Tripy's 3 main components are:

1. **Backend**: Interfaces with MLIR-TRT:

    - The **Compiler** compiles tensorrt-dialect MLIR to an MLIR-TRT executable

    - The **Executable** wraps an MLIR-TRT executable to allow for inference with {class}`nvtripy.Tensor`s.

2. **Trace**: Represents the computation graph as [`TraceTensor`s](source:/nvtripy/trace/tensor.py)
    and [`TraceOp`s](source:/nvtripy/trace/ops/base.py) and can **lower** to tensorrt-dialect MLIR.

3. **Frontend**: Exposes functional-style Python APIs that work with {class}`nvtripy.Tensor`s.
