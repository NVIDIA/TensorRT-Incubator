#ifndef MLIR_TENSORRT_DIALECT_CUDA_IR_CUDAATTRIBUTES
#define MLIR_TENSORRT_DIALECT_CUDA_IR_CUDAATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.td"

//===----------------------------------------------------------------------===//
// CUDA Dialect Enums
//===----------------------------------------------------------------------===//

// Enums must be included here since every enum has a corresponding attribute.
include "mlir-tensorrt/Dialect/CUDA/IR/CUDAEnums.td"

//===----------------------------------------------------------------------===//
// CUDA Dialect Attribute Classes and Parameters
//===----------------------------------------------------------------------===//

class CUDA_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CUDA_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // MLIR_TENSORRT_DIALECT_CUDA_IR_CUDAATTRIBUTES
