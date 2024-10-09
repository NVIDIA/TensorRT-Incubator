add_mlir_tensorrt_library(MLIRTensorRTRuntimeToExecutor
  TensorRTRuntimeToExecutor.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRReconcileUnrealizedCasts
  MLIRRewrite
  MLIRTensorRTConvertToExecutorCommon
  MLIRTensorRTCUDADialect
  MLIRTensorRTDialect
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorUtils
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTransforms
  MLIRTransformUtils
)
