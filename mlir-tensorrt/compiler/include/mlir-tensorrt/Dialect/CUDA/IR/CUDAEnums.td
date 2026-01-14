//===- CUDAEnums.td -------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_DIALECT_CUDA_IR_CUDAENUMS_TD
#define MLIR_TENSORRT_DIALECT_CUDA_IR_CUDAENUMS_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.td"

//===----------------------------------------------------------------------===//
// Helper classes for defining CUDA enum attributes
//===----------------------------------------------------------------------===//

class CUDA_I32EnumAttr<string name, string description,
                       list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
}

class CUDA_EnumAttr<IntEnumAttr intEnumAttr, string mnemonic,
                    list<Trait> traits = []>
    : EnumAttr<CUDA_Dialect, intEnumAttr, mnemonic, traits> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// CUDA compiled module enums
//===----------------------------------------------------------------------===//

def CUDA_CompiledModuleKind : CUDA_I32EnumAttr<
  "CompiledModuleKind",
  "Kind of compiled module stored in cuda.compiled_module",
  [
    I32EnumAttrCase<"PTX", 0>,
    I32EnumAttrCase<"LLVMIR", 1>
  ]> {
  let cppNamespace = "::mlir::cuda";
  let genSpecializedAttr = 0;
}

def CUDA_CompiledModuleKindAttr : CUDA_EnumAttr<CUDA_CompiledModuleKind, "compiled_module_kind"> {}

#endif // MLIR_TENSORRT_DIALECT_CUDA_IR_CUDAENUMS_TD
