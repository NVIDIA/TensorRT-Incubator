#ifndef MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_TENSORRTRUNTIMETYPES_TD
#define MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_TENSORRTRUNTIMETYPES_TD

include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeDialect.td"

include "mlir/IR/AttrTypeBase.td"

def TensorRTRuntime_Engine :TypeDef<TensorRTRuntime_Dialect, "Engine", []> {
  let mnemonic = "engine";
  let description = [{
    An opaque object which represents an `nvinfer1::ICudaEngine` instance.
  }];
}

def TensorRTRuntime_Context :TypeDef<TensorRTRuntime_Dialect,
    "ExecutionContext", []> {
  let mnemonic = "context";
  let description = [{
    An opaque object which represents an `nvinfer1::IExecutionContext` instance.
  }];
}

#endif // MLIR_TENSORRT_DIALECT_TENSORRTRUNTIME_IR_TENSORRTRUNTIMETYPES_TD
