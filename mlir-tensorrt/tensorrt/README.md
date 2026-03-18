# TensorRT MLIR Dialect

The `tensorrt` dialect is an MLIR dialect that precisely models the
[TensorRT](https://developer.nvidia.com/tensorrt) builder API. It lets you
represent, validate, optimize, and translate TensorRT networks entirely within
MLIR before emitting a serialized TensorRT engine.

This component is designed to be self-contained: it can be used independently in
any MLIR-based compiler that wants to target TensorRT, without pulling in the
rest of MLIR-TensorRT.

## What it does

1. **Model** — Every TensorRT layer type has a corresponding MLIR operation
   (e.g. `tensorrt.convolution`, `tensorrt.element_wise`,
   `tensorrt.matrix_multiply`). Operations carry the same parameters and
   constraints as the C++ TensorRT API.

2. **Validate** — The dialect's verifiers enforce TensorRT's type, rank, and
   attribute constraints at IR-construction time rather than at engine-build
   time, providing earlier and more actionable error messages.

3. **Transform** — A set of optimization passes simplifies and canonicalizes
   the IR before translation (broadcast elimination, transpose/reshape
   folding, activation raising, int8 legalization, and more).

4. **Translate** — A translation pass builds an `nvinfer1::INetworkDefinition`
   from the MLIR, configures an optimization profile, and invokes the TensorRT
   builder to produce a serialized engine.

## Operations

Operations are generated from the TensorRT API header `NvInfer.h`. The main
categories are:

| Category | Examples |
|---|---|
| Structural | `tensorrt.module`, `tensorrt.call`, `tensorrt.call_alloc` |
| Elementwise / Unary | `tensorrt.element_wise`, `tensorrt.unary`, `tensorrt.activation`, `tensorrt.cast` |
| Reduction | `tensorrt.reduce`, `tensorrt.argmin`, `tensorrt.argmax`, `tensorrt.top_k` |
| Convolution / Pooling | `tensorrt.convolution`, `tensorrt.deconvolution`, `tensorrt.pooling` |
| Linear Algebra | `tensorrt.matrix_multiply`, `tensorrt.einsum` |
| Shape Manipulation | `tensorrt.shuffle`, `tensorrt.reshape`, `tensorrt.transpose`, `tensorrt.expand_rank`, `tensorrt.collapse_rank`, `tensorrt.broadcast`, `tensorrt.slice`, `tensorrt.concatenation`, `tensorrt.padding` |
| Gather / Scatter | `tensorrt.gather`, `tensorrt.gather_nd`, `tensorrt.gather_elements`, `tensorrt.scatter_nd`, `tensorrt.scatter_elements` |
| Normalization | `tensorrt.normalization`, `tensorrt.softmax`, `tensorrt.ragged_softmax` |
| Quantization | `tensorrt.quantize`, `tensorrt.dequantize`, `tensorrt.dynamic_quantize` |
| Constant / Fill | `tensorrt.constant`, `tensorrt.linspace`, `tensorrt.random_uniform`, `tensorrt.random_normal` |
| Resize | `tensorrt.resize_nearest`, `tensorrt.resize_linear`, `tensorrt.resize_cubic` |
| Control Flow | `tensorrt.if`, `tensorrt.while`, `tensorrt.for`, `tensorrt.condition`, `tensorrt.yield` |
| Plugin | `tensorrt.opaque_plugin` |
| Misc | `tensorrt.select`, `tensorrt.one_hot`, `tensorrt.non_zero`, `tensorrt.identity`, `tensorrt.parametric_relu`, `tensorrt.shape`, `tensorrt.assertion` |

## Type System

TensorRT tensors are represented as ranked MLIR tensors with restrictions that
match TensorRT's requirements:

- **Ranks**: 0–8
- **Element types**: `i1`, `i8`, `i32`, `i64`, `f16`, `bf16`, `f32`
- **Quantized types**: `TensorRT_QuantizedI8`, `TensorRT_F8`, `TensorRT_I4`,
  `TensorRT_F4`
- **Shape tensors**: 1-D static `i32` tensors (used for dynamic shape
  computations inside the network)

Dynamic shapes are modeled with the `tensorrt.shape_profile` function argument
attribute, which specifies min/opt/max bounds for each dynamic dimension.

## Passes

### Optimization Passes

| Pass | Description |
|---|---|
| `tensorrt-raise-activations` | Recognize and raise fused activation patterns (e.g. GELU) |
| `tensorrt-raise-normalizations` | Recognize and raise normalization patterns |
| `tensorrt-broadcast-elimination` | Remove or absorb redundant broadcasts |
| `tensorrt-transpose-elimination` | Fold away unnecessary transposes |
| `tensorrt-reshape-elimination` | Fold away unnecessary reshapes |
| `tensorrt-transpose-reshape-elimination` | Combined transpose/reshape/shuffle simplification |
| `tensorrt-expand-ops` | Expand extension ops into lower-level TensorRT ops |
| `tensorrt-legalize-int8` | Legalize int8 and QDQ operations for TensorRT compatibility |
| `tensorrt-apply-wars` | Apply workarounds for known TensorRT limitations |
| `tensorrt-infer-plugin-shapes` | Infer output shapes for custom plugin operations |

### Translation Pass

| Pass | Description |
|---|---|
| `translate-tensorrt-to-engine` | Build TensorRT engines from MLIR functions and attach the serialized engine as an attribute |

### Pipelines

Two pre-built pass pipelines combine the above passes in a useful order:

- **Simplification pipeline** — broadcast elimination, transpose/reshape
  elimination, and normalization raising.
- **Transformation pipeline** — simplification plus workarounds, op expansion,
  and int8 legalization.

## Translation to TensorRT Engines

Translation is implemented through two key interfaces:

- **`TensorRTOpInterface`** — marks an operation as a TensorRT layer.
- **`TensorRTEncodingOpInterface`** — provides an `encodeOp` method that maps
  an MLIR operation to TensorRT API calls via the `NvInferNetworkEncoder`.

The translation flow:

1. For each function in a `tensorrt.module`, create an
   `nvinfer1::INetworkDefinition`.
2. Walk the function body. For each operation, call `encodeOp` to add the
   corresponding TensorRT layer and map MLIR SSA values to `nvinfer1::ITensor*`
   handles.
3. Configure an `IBuilderConfig` with the requested precision flags (FP16,
   INT8, FP8, BF16, etc.) and the optimization profile derived from
   `tensorrt.shape_profile` attributes.
4. Build the serialized engine and attach it to the function as a
   `tensorrt.engine` attribute.

The resulting engine can be extracted and executed with TensorRT's runtime API,
or further processed by downstream MLIR-TensorRT compiler stages.

## Directory Layout

```
tensorrt/
├── include/mlir-tensorrt-dialect/
│   ├── TensorRT/
│   │   ├── IR/                 # Dialect, ops, enums, attributes (TableGen + headers)
│   │   ├── Transforms/         # Pass declarations (TableGen)
│   │   ├── Target/             # Encoding interface declarations
│   │   └── Utils/              # Utility headers
│   └── Target/                 # Translation pass, network encoder API
│       └── TensorRTEncodingOpInterface/
├── lib/
│   ├── TensorRT/
│   │   ├── IR/                 # Dialect and op implementations, type inference
│   │   ├── Transforms/         # Pass implementations
│   │   ├── Target/             # Per-op encoding interface implementations
│   │   └── Utils/              # NvInfer adaptors, shape utilities, plugin helpers
│   ├── Target/                 # TranslateToTensorRT.cpp
│   │   └── TensorRTEncodingOpInterface/  # NetworkEncoder.cpp
│   ├── Analysis/               # TensorKindAnalysis
│   └── CAPI/                   # C API bindings
├── test/
│   ├── Dialect/TensorRT/       # Canonicalization tests
│   └── Target/TensorRT/        # Translation tests (per-op, TRT10, strongly typed, plugins)
└── tools/
    ├── tensorrt-opt/            # MLIR pass runner for the TensorRT dialect
    └── tensorrt-tblgen/         # Custom TableGen backend for encoding
```

## Example

```mlir
tensorrt.module @my_network {
  func.func @main(%arg0: tensor<1x3x224x224xf32> {tensorrt.shape_profile = #tensorrt.shape_profile<min=[1,3,224,224], opt=[1,3,224,224], max=[4,3,224,224]>})
      -> tensor<?x1000xf32> {
    %w = tensorrt.constant dense<...> : tensor<64x3x7x7xf32>
    %conv = tensorrt.convolution {
        pre_padding = array<i64: 3, 3>,
        post_padding = array<i64: 3, 3>,
        stride = array<i64: 2, 2>
    } in(%arg0 : tensor<1x3x224x224xf32>) kernel(%w : tensor<64x3x7x7xf32>)
      -> tensor<?x64x112x112xf32>
    %relu = tensorrt.activation {activationType = #tensorrt.activation_type<kRELU>}
      %conv : tensor<?x64x112x112xf32>
    // ... remaining layers ...
  }
}
```

Running `tensorrt-opt --translate-tensorrt-to-engine` on the above produces a
function with a `tensorrt.engine` attribute containing the serialized engine
bytes.
