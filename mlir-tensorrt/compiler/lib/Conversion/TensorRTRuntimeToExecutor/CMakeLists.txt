add_mlir_tensorrt_library(MLIRTensorRTRuntimeToExecutor
  TensorRTRuntimeToExecutor.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTConvertToExecutorCommon
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
