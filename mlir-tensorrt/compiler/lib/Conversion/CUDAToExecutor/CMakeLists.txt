add_mlir_tensorrt_library(MLIRTensorRTCUDAToExecutor
  CUDAToExecutor.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTConvertToExecutorCommon
    MLIRTensorRTCUDADialect
    MLIRTensorRTExecutorDialect
    MLIRTensorRTExecutorRuntimeAPI
    MLIRTensorRTExecutorUtils
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRRewrite
)
