add_mlir_tensorrt_library(MLIRTensorRTRuntimeToExecutor
  TensorRTRuntimeToExecutor.cpp

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorPasses
    MLIRTensorRTCUDADialect
    MLIRTensorRTTensorRTDialect
    MLIRTensorRTExecutorDialect
    MLIRTensorRTExecutorUtils
    MLIRTensorRTTensorRTRuntimeDialect
  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRReconcileUnrealizedCasts
    MLIRRewrite
    MLIRTransforms
    MLIRTransformUtils
)
