add_mlir_tensorrt_library(MLIRTensorRTPlanToExecutor
  PlanToExecutor.cpp

  LINK_LIBS PUBLIC
  MLIRFuncTransforms
  MLIRTensorRTExecutorDialect
  MLIRTensorRTPlanDialect
  MLIRTransforms
  MLIRSCFTransforms  
  )
