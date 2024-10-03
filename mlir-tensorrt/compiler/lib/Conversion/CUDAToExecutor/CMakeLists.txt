add_mlir_tensorrt_library(MLIRTensorRTCUDAToExecutor
  CUDAToExecutor.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRRewrite
  MLIRTensorRTConvertToExecutorCommon
  MLIRTensorRTCUDADialect
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorRuntimeAPI
  MLIRTensorRTExecutorUtils
)
