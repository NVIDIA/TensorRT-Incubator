add_mlir_tensorrt_library(MLIRTensorRTTensorRTToTensorRTRuntime
  TensorRTToTensorRTRuntime.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorUtils
  MLIRTensorRTTensorRTDialect
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTensorRTAnalysis
  MLIRRewrite
  MLIRTensorRTConvertToExecutorCommon
)
