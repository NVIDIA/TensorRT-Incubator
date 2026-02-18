#ifndef MLIR_TENSORRT_COMMON_CONVERSION_PASSES
#define MLIR_TENSORRT_COMMON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertToLoops : Pass<"convert-to-loops"> {
  let summary = "Convert a LoopLikeOpInterface to loops";
  let description = [{
    This pass converts a LoopLikeOpInterface to loops.
  }];

  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::scf::SCFDialect",
  ];
}

#endif // MLIR_TENSORRT_COMMON_CONVERSION_PASSES
