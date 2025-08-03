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
  MLIRTensorRTTensorRTDialect
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorUtils
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTransforms
  MLIRTransformUtils
)
