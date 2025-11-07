#ifndef MLIR_TENSORRT_DIALECT_CUDA_IR_CUDADIALECT_TD
#define MLIR_TENSORRT_DIALECT_CUDA_IR_CUDADIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CUDA Dialect
//===----------------------------------------------------------------------===//

def CUDA_Dialect : Dialect {
  let name = "cuda";
  let summary = "A dialect that models CUDA API";
  let description = [{
    "CUDA dialect models CUDA types and operations. These types and operations
    will be used by the lower level `Executor` dialect.
  "}];
  let cppNamespace = "::mlir::cuda";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // MLIR_TENSORRT_DIALECT_CUDA_IR_CUDADIALECT_TD
