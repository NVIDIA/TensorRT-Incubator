add_mlir_tensorrt_library(MLIRTensorRTCUDAToExecutor
  CUDAToExecutor.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRTensorRTExecutorRuntimeAPI
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorUtils
  MLIRTensorRTCUDADialect
  MLIRRewrite
)
