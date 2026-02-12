add_mlir_tensorrt_library(MLIRTensorRTCUDAToExecutor
  CUDAToExecutor.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorPasses
    MLIRTensorRTCUDADialect
    MLIRTensorRTExecutorDialect
    MLIRTensorRTExecutorRuntimeAPI
    MLIRTensorRTExecutorUtils
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRRewrite
)
