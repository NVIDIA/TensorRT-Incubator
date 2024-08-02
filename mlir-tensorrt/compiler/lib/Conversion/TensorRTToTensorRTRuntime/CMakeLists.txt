add_mlir_tensorrt_library(MLIRTensorRTTensorRTToTensorRTRuntime
  TensorRTToTensorRTRuntime.cpp

  LINK_LIBS
  MLIRTensorRTExecutorUtils
  MLIRTensorRTDialect
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTensorRTAnalysis
  MLIRRewrite
  MLIRTensorRTConvertToExecutorCommon
)
