add_mlir_tensorrt_library(MLIRTensorRTPlanToExecutor
  PlanToExecutor.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorDialect
    MLIRTensorRTPlanDialect
  MLIR_LIBS PUBLIC
    MLIRFuncTransforms
    MLIRTransforms
    MLIRSCFTransforms
  )
