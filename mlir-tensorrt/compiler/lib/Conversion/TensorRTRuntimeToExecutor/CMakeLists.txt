add_mlir_tensorrt_library(MLIRTensorRTRuntimeToExecutor
  TensorRTRuntimeToExecutor.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRTensorRTExecutorDialect
  MLIRTensorRTCUDADialect
  MLIRTensorRTExecutorUtils
  MLIRTensorRTConvertToExecutorCommon
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTensorRTDialect
  MLIRReconcileUnrealizedCasts
  MLIRRewrite
  MLIRTransforms
  MLIRTransformUtils
)
