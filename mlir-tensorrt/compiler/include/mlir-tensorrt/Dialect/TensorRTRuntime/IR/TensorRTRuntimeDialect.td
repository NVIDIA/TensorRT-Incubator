#ifndef TENSORRTRUNTIMEDIALECT_TD
#define TENSORRTRUNTIMEDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TensorRTRuntime Dialect
//===----------------------------------------------------------------------===//

def TensorRTRuntime_Dialect : Dialect {
  let name = "trtrt";
  let summary = "A dialect that models the TensorRT Runtime API";
  let description = [{

    Contains a collection of operations and attributs that model how TensorRT
    compilation artifacts are loaded and executed at runtime.

    This dialect is not meant to be opinionated in terms of how the operations
    are lowered; there could be multiple possible lowering paths.

  }];
  let cppNamespace = "::mlir::trtrt";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
  let dependentDialects = [
    "::mlir::cuda::CUDADialect"
  ];
}

#endif // TENSORRTRUNTIMEDIALECT_TD
