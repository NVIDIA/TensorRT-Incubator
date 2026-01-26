//===- Passes.td ----------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_MLIR_KERNEL_CONVERSION_PASSES
#define INCLUDE_MLIR_KERNEL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertKernelToLLVM : Pass<"convert-kernel-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Converts Kernel IR to LLVM (MLIR) IR";

  let description = [{
    This pass converts Kernel dialect operations to LLVM dialect operations,
    enabling the execution of GPU kernels through the CUDA Runtime API.

    The pass performs the following transformations:

    1. Converts `kernel.call` operations into LLVM IR calls to the CUDA Runtime
       function `cudaLaunchKernelExC`, which is used to launch GPU kernels.

    2. Inlines the contents of `gpu.module` operations into the parent module.
       Before inlining, it updates all symbol references to kernel functions by
       removing the `gpu.module` scope from nested symbol references (e.g.,
       `@gpu_module::@kernel_func` becomes `@kernel_func`).

    3. Constructs the required `cudaLaunchConfig_t` structure on the stack,
       including grid dimensions, block dimensions, shared memory size, and
       stream handle.

    4. Prepares kernel parameters by:
       - Scalarizing memref operands into their constituent parts
       - Allocating stack storage for each scalar parameter
       - Creating an array-of-pointers to these stack-allocated parameters

    Preconditions:
    - The `kernel.call` operation should be bufferized (no results)
    - All operations within `gpu.module` must already be lowered to LLVM dialect
  }];

  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// EliminateIndexTypes
//===----------------------------------------------------------------------===//

def EliminateIndexTypes : Pass<"kernel-eliminate-index-types"> {
  let summary = "Eliminate index types from the device code";
  let description = [{
    This pass is meant to be run after the `lower-affine` pass. It eliminates
    `index` types by replacing them with the specified bitwidth integer
    type. The default replacement is `i32`.

    This pass uses MLIR's dialect conversion infrastructure.

    For sake of breviy, assume `index` is to be replaced with `i32`.
    The type conversions are as follows:
    - `index` -> `i32`
    - `tensor<...xindex>` -> `tensor<...xi32>`
    - `memref<...xindex>` -> `memref<...xi32>`

    The conversion patterns handle the following cases:

    - SCF and Func dialect structural type conversion patterns
      are populated. Other types of function-like ops or region operations
      are not handled.

    - `arith.index_cast` and `arith.index_castui` are handled by either replacing with the input
      (if adaptor value has same type as result) or using the appropriate
      Arith dialect cast operation (e.g. `arith.extsi` or `arith.trunci`
      or`arith.extui`).

    - `arith.constant ... : index` is handled by converting the attribute to the appropriate
      integer type.

    The conversion is a "full" conversion -- any remaining index types are
    considered an error.
  }];

  let options = [Option<"indexBitwidth", "index-bitwidth", "unsigned", "64",
                        "Bitwidth of the index type">];
}

//===----------------------------------------------------------------------===//
// ConvertComplexToStandardExt
//===----------------------------------------------------------------------===//

def ConvertComplexToStandardExt : Pass<"convert-complex-to-std-ext"> {
  let summary = "Convert complex dialect to standard operations";
  let description = [{
    This pass is meant to be run after `convert-complex-to-standard`. The goal
    is to entirely eliminate `complex<f32|f64>` and `memref<...xcomplex<f32|f64>>`
    types from the device code since some backends do not support
    complex types or tuple/struct types to represent complex numbers.

    The pass enforces a specific storage ABI for all complex-valued shaped
    types (memref and tensor) and scalar types within `gpu.module` operations. Only
    `complex<f32>` and `complex<f64>` are supported. These types are converted
    to `i64` and `i128` respectively where the real component is stored into the
    lower bits and the imaginary component is stored into the higher bits.
    Operations such as `complex.constant` and `complex.create` are converted
    to the equivalent operations to pack the data into an integer value.
    Similarly, `memref<...xcomplex<f32>>` and `memref<...xcomplex<f64>>` are
    converted to `memref<...xi64>` and `memref<...xi128>` respectively. The same
    transformation is also applied to tensor element types.

    All corresponding operations like `memref.load|store`,
    `tensor.extract|insert` and so on are explicitly converted as long as they
    are nested within a `gpu.module` operation. Outside of `gpu.module` operations,
    operations are only converted if they are in the `convert-op-generically` list.
    Typically, this should include operations `kernel.call` and `memref.cast`
    so that the host-side launch operations are correctly converted.

    Outside of the specific operations that this pass knows how to convert from
    the complex, memref, tensor, bufferization, linalg, and arith dialects,
    unknown operations will be left unmodified. The
    `builtin.unrealized_conversion_cast` op is used to cast types at the
    boundary with unknown operations.

    The `convert-op-generically` option is a list of operation names where
    generic conversion logic should be applied, regardless of whether the
    operation is nested within a `gpu.module` operation. The pass user should
    guarantee that operations in this list are legal to convert by swapping any
    `complex<...>` or `memref<...xcomplex<...>>` types with the converted
    integer types.
  }];
  let options = [
    ListOption<"convertOpGenerically", "convert-op-generically", "std::string",
      "List of operation names where generic conversion logic should be applied."
    >
  ];

  let dependentDialects = [
    "::mlir::executor::ExecutorDialect"
  ];
}

#endif // INCLUDE_MLIR_KERNEL_CONVERSION_PASSES
