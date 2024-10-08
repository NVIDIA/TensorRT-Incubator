# Using Python APIs of MLIR Dialects


## Choosing The Right Dialect(s)

The first step is to choose the correct MLIR ops from one or more dialects:

- [stablehlo](https://github.com/openxla/stablehlo/blob/main/docs/spec.md) dialect is
    usually the first choice as it covers most of the computing ops.

    **You can refer to [jax's implementations](https://github.com/google/jax/blob/059fdaf1554ff508db5e267b884d7d47f583fe8a/jax/_src/lax/)**
    **before looking for other dialects.**

- For some ops like tensor allocation, initialization, copy or custom ops, we need ops from other
    [MLIR dialects](https://mlir.llvm.org/docs/Dialects/); the most helpful ones includes:
    `arith`, `tensor`, `bufferization`, and `linalg`.

- Sometimes we may need to mix ops from different dialects to implement one operation; for example, we may require
    `arith.constant` to declare a single constant value, and then `tensor.empty` or `bufferization.alloc_tensor`
    to allocate an output tensor, and finally `linalg.fill` or other ops to perform the operation we want.


## Python Source Code

The python source code is auto-generated according to `.td` files.

1. Check the `.td` file for the op description:
    - [stablehlo.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td)
    - Other dialects' `.td` files are under `mlir-tensorrt/third_party/llvm-project/mlir/include/mlir/Dialect/`

2. Check the generated python source code for detailed API:
    - stablehlo (requires the built package): `stablehlo/python-build/tools/stablehlo/python_packages/stablehlo/mlir/dialects/_stablehlo_ops_gen.py`
    - other dialects: `stablehlo/python-build/tools/mlir/python/dialects/`
    - most importantly, the `ir` module: `stablehlo/python-build/tools/stablehlo/python_packages/stablehlo/mlir/ir.py` and
        `stablehlo/python-build/tools/stablehlo/python_packages/stablehlo/mlir/_mlir_libs/_mlir/ir.pyi`


## Example Usage

We can find some examples in tests, but not all of the ops are covered.

- [stablehlo](https://github.com/openxla/stablehlo/tree/main/stablehlo/integrations/python/tests): almost no test code for its ops' python APIs
- other dialects: `mlir-tensorrt/third_party/llvm-project/mlir/test/python/dialects`

*TIP: Always search for the op's python API within `*.py` files under `mlir-tensorrt` and `stablehlo`.*
