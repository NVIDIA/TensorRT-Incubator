add_mlir_tensorrt_library(MLIRTensorRTTensorRTToTensorRTRuntime
  TensorRTToTensorRTRuntime.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorUtils
  MLIRTensorRTDialect
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTensorRTAnalysis
  MLIRRewrite
  MLIRTensorRTConvertToExecutorCommon
)
